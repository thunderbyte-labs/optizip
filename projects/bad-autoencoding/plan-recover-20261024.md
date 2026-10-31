# Plan recovery — crash système `hera` du 2026-10-24

> **Document privé. Ne pas committer ni pousser sur GitHub.**
> À lire dans une conversation séparée. Objectif : analyse forensic complète du
> crash, puis mise en place des garde-fous pour que le serveur ne recrashe plus.

---

## 0. Objet et principe

Le serveur `hera` (2× RX 9070 XT `gfx1201`, Ryzen 9 9950X3D, 248 Go RAM,
Ubuntu 24.04, kernel 7.0.0-31, ROCm 7.2.1) a freeze le **2026-10-24 vers
00:50 UTC**, après **44 jours d'uptime**. L'utilisateur a dû faire un
`Alt+PrtSc+REISUB` (sync + remount RO + reboot), ce qui a interrompu les
services hébergés (nœud Hyperliquid, fathom, observateur, etc.).

Le but n'est **pas** de relancer le training tout de suite. Le but est :

1. comprendre la cause (ou au minimum exclure les causes matérielles) ;
2. rendre le serveur robuste aux charges GPU ;
3. seulement ensuite, reprendre les runs GPU dans un cadre contrôlé.

**Le container seul ne protège pas d'un hang du driver GPU** (le noyau et le
driver `amdgpu` sont partagés). Il faut combiner : isolation process/cgroup,
serialisation des charges GPU, watchdog hôte, limites de puissance/thermique,
et configuration kernel/driver saine.

---

## 1. Chronologie factuelle

| Heure UTC | Événement |
|---|---|
| 2026-09-11 04:51 | Boot précédent (uptime 44 j) |
| 2026-10-23 10:20 | `llama-server` traite un prompt de ~55 000 tokens (GPU intensif) |
| 2026-10-23 11:39:45 | `llama-server` arrêté proprement (PID 536116) |
| 2026-10-23 13:00–14:45 | Tentatives FSDP/probes sur `hera` : OOM mémoire, pas de crash système |
| 2026-10-24 00:40:19–00:41:06 | SysRq `r`/`k` refusés (« This sysrq operation is disabled ») ; HELP SysRq affiché à 00:40:49 |
| 2026-10-24 00:49:37 | Dernier log applicatif normal (observateur) |
| 2026-10-24 00:50:09–00:50:10 | Nouvelles tentatives SysRq refusées |
| 2026-10-24 00:50:11 | `sysrq: Emergency Sync` puis `complete` |
| 2026-10-24 00:50:12 | `sysrq: Emergency Remount R/O` ; `/home` et `/` remontés en lecture seule |
| 2026-10-24 00:51:29 | Nouveau boot (kernel 7.0.0-31) |
| 2026-10-24 00:51:31 | `llama-server.service` redémarre automatiquement (`Restart=always`) |
| 2026-10-24 02:47:20 | Après un `kill` manuel, systemd redémarre `llama-server` en 10 s |
| 2026-10-24 ~03:00 | L'utilisateur fait `systemctl stop` + `disable` ; GPU libérés |

**Point important : `llama-server` n'était PAS en cours au moment du crash de
00:50.** Il avait été arrêté à 11:39 la veille. Le crash n'est donc pas
directement imputable au service LLM, même si les charges GPU antérieures
(notre training, le prompt 55k tokens) ont pu laisser le driver dans un état
dégradé.

---

## 2. Constats déjà établis (lecture seule, avant ce plan)

- `journalctl --list-boots` : boot `-1` du 2026-09-11 04:51 au 2026-10-24
  00:50:12 ; boot `0` depuis 00:51:29.
- **Aucun message `amdgpu`, `kfd`, `GPU reset`, `ring timeout`, `MCE`,
  `thermal`, `OOM`, `PCIe AER` ou `panic`** dans le boot précédent, ni dans
  `/var/log/kern.log` autour de 00:40–00:51. Les dernières lignes noyau sont
  des `UFW BLOCK` puis les SysRq.
- Le noyau était **vivant** : `s` (sync) et `u` (remount RO) ont fonctionné ;
  le userspace / la console / l'affichage était figé.
- `Xorg.0.log.old` date du **2026-10-23 13:17**, pas du crash ; le log Xorg de
  la session crashée a été écrasé au reboot. `Xorg.0.log` courant date de
  01:20 le 24.
- `sar` (sysstat, intervalle 10 min) n'a **aucun échantillon entre 00:07 et
  00:50** : pas de données CPU/RAM/IO du moment du crash. `sa23` s'arrête à
  00:07 ; `sa24` commence à 01:30 (nouveau boot).
- `/var/crash/kexec_cmd` existe (kdump configuré) mais **aucun kdump** n'a été
  produit (ce n'était pas un panic noyau). Pas d'entrée `pstore`.
- Cmdline noyau actuelle :
  ```
  BOOT_IMAGE=/vmlinuz-7.0.0-31-generic root=UUID=... ro quiet splash
  amdgpu.runpm=0 amdgpu.lockup_timeout=1000 amdgpu.gpu_recovery=1
  crashkernel=2G-4G:320M,... crashkernel=512M vt.handoff=7
  ```
  **`amdgpu.lockup_timeout=1000` = 1 seconde** (défaut : 10 000 ms). Très
  agressif pour des kernels GPU longs. **Suspect n°1.**
  Il y a aussi **deux `crashkernel=`** (à nettoyer).
- `llama-server.service` : `Restart=always`, `RestartSec=5`, `enabled`,
  `HIP_VISIBLE_DEVICES=0,1`, modèle Qwen3.8-27B Q5, **`--ctx-size 131072`**,
  `-ngl 99`, `--tensor-split 1,1`, `--batch-size 1024`, 16 threads. Le KV cache
  q8_0 sur 131k de contexte est énorme pour 2×16 Go.
- `hera-observateur` ne logge que `gpu0` et seulement l'utilisation
  (`gpu0=0.0%`), sans temp/power/VRAM/GPU1.
- RAM actuelle : 248 Go, ~208 Go disponibles, 2 Go libres (le reste en cache) ;
  load average 18 avant l'arrêt de llama.
- Après arrêt/désactivation de `llama-server`, les deux dGPU sont libres :
  **15,82 / 15,85 GiB libres**.

---

## 3. Hypothèses, classées

1. **Reset GPU déclenché par `amdgpu.lockup_timeout=1000`** pendant un kernel
   long (training, prompt 55k, MIOpen), suivi d'un hang du driver/affichage.
   Le noyau n'a pas eu le temps de flusher un log `amdgpu` avant le REISUB.
   C'est l'hypothèse la plus cohérente avec l'absence de log et le freeze.
2. **Freeze Xorg / iGPU / compositeur** : le noyau alive, le clavier/console
   figé, les VT (`Ctrl+F3`) inopérants. L'iGPU Raphael pilote l'affichage ;
   un hang iGPU/Mesa peut figer le desktop sans log noyau.
3. **Bug kernel 7.0.0-31 / uptime 44 jours** : fuite ou régression driver non
   identifiable sans plus de logs.
4. **Transitoire alimentation/thermique** : peu probable (pas de MCE, pas de
   log thermal, le noyau est resté vivant), mais à vérifier côté PSU/câbles.
5. **Pression RAM/IO** : pas d'OOM au moment du crash ; le V8 OOM de Brave du
   22/10 est antérieur. À confirmer avec `sar`/journal.
6. **État dégradé du driver après les OOM GPU du 23/10** : les OOM PyTorch ne
   sont pas anodins ; ils peuvent laisser le driver dans un état instable. Une
   règle : **reboot après tout OOM/hang GPU**, ne pas enchaîner les runs.

---

## 4. Forensic à compléter (à exécuter dans la conversation séparée)

> Certaines commandes demandent `sudo`. Ne rien modifier avant d'avoir collecté
> et sauvegardé les sorties dans `~/forensic-20261024/`.

### 4.1 Boot précédent / noyau

```bash
mkdir -p ~/forensic-20261024
journalctl -b -1 -o short-iso --no-pager > ~/forensic-20261024/journal-boot-1.txt
journalctl -b -1 -k -o short-iso --no-pager > ~/forensic-20261024/kernel-boot-1.txt
journalctl -b -1 --since "2026-10-24 00:30" --until "2026-10-24 00:51" \
  -o short-iso --no-pager > ~/forensic-20261024/crash-window.txt
journalctl -b -1 | grep -m3 "Linux version"    # version exacte du boot crashé
journalctl --list-boots --no-pager
last -x | head -20
cat /proc/cmdline
```

### 4.2 Logs noyau/périphériques (le journal peut être incomplet)

```bash
# Le kern.log courant contient-il encore la fin du boot précédent ?
grep -aE "2026-10-24T00:(3[0-9]|4[0-9]|5[0-2])" /var/log/kern.log > ~/forensic-20261024/kern-crash.txt
grep -aE "2026-10-24T00:(3[0-9]|4[0-9]|5[0-2])" /var/log/syslog > ~/forensic-20261024/syslog-crash.txt
sudo dmesg -T > ~/forensic-20261024/dmesg-current.txt
sudo journalctl -b -1 -k | grep -iE "amdgpu|kfd|ring|timeout|reset|mce|edac|aer|pcie|thermal|throttle|power|hung|blocked|oom|panic|BUG" > ~/forensic-20261024/kernel-errors.txt
sudo ls -l /sys/fs/pstore; sudo cat /sys/fs/pstore/* 2>/dev/null
sudo ls -l /var/crash; sudo cat /var/crash/kexec_cmd 2>/dev/null
```

### 4.3 Xorg / affichage

```bash
ls -lh /var/log/Xorg.*.log* ~/.xsession-errors* ~/.local/share/xorg/ 2>/dev/null
grep -iE "amdgpu|gpu|hang|reset|error|timeout|drm" /var/log/Xorg.0.log.old | tail -80
grep -iE "amdgpu|gpu|hang|reset|error|timeout|drm" /var/log/Xorg.0.log | tail -80
# Si le log du crash a été écrasé, ajouter une copie au boot :
#   cp /var/log/Xorg.0.log /var/log/Xorg.0.log.$(date +%Y%m%d-%H%M)
```

### 4.4 CPU/RAM/IO au moment du crash

```bash
sar -u -r -q -b -d -f /var/log/sysstat/sa23 | tail -40
sar -u -r -q -b -d -f /var/log/sysstat/sa24 | head -40
free -h; systemd-cgtop -m --iterations=1 2>/dev/null | head -20
```

### 4.5 Matériel / alimentation / stockage

```bash
sudo smartctl -a /dev/nvme0n1
sudo nvme smart-log /dev/nvme0n1
sudo fsck -n /dev/nvme0n1p2 /dev/nvme0n1p4    # lecture seule ; ne pas réparer à chaud
sudo dmidecode -t memory
sudo dmidecode -t system | grep -iE "product|manufacturer|power"
lspci -vv | grep -A20 -E "VGA|Display|PCI bridge" | grep -iE "LnkSta|AER|Kernel driver"
sudo sensors; rocm-smi --showtemp --showpower --showclocks --showdriverversion
dkms status; uname -a; apt list --installed 2>/dev/null | grep linux-image
```

### 4.6 Services et charge GPU avant le crash

```bash
journalctl -b -1 -u llama-server --no-pager | tail -40
journalctl -b -1 -u hera-observateur-tirer --no-pager | tail -30
journalctl -b -1 -u 'hera-node-watchdog' --no-pager | tail -30
journalctl -b -1 -u 'fathom*' --no-pager | tail -30
cat /home/hera-observateur/rapports/rapport_gpu.csv 2>/dev/null | tail -120
coredumpctl list --no-pager 2>/dev/null | tail -20
```

### 4.7 Ce qu'il faut chercher

- Un `amdgpu: GPU reset(1)` / `ring gfx timeout` / `MEC` / `RLC` / `SDMA`
  autour de 00:39–00:50, même hors journal (kern.log, pstore).
- Un `mce` / `edac` / `AER` / `thermal` / `throttle`.
- Un `oom-kill` ou une saturation RAM.
- Un crash Xorg (`(EE)`, `Failed to idle channel`, `drmWaitVBlank`).
- La température/puissance GPU dans les minutes précédentes.
- Le load average / IO wait (sar).
- La version noyau du boot crashé et les mises à jour récentes.

---

## 5. Plan de prévention

### 5.1 Quick wins (à faire en premier)

1. **Corriger `amdgpu.lockup_timeout=1000` → `10000` (ou `20000`)** dans
   `/etc/default/grub`, nettoyer le double `crashkernel=`, garder
   `amdgpu.runpm=0 amdgpu.gpu_recovery=1`, puis `sudo update-grub` et reboot
   dans une fenêtre de maintenance. C'est le changement le plus susceptible
   d'éviter un reset GPU sous charge.
2. **Laisser `llama-server` désactivé** tant que le garde-fou n'est pas en
   place. Si le service est nécessaire :
   - le déplacer sur le **DGX** (121 Go, plus adapté à un 27B) ; ou
   - réduire `--ctx-size` à **16 384** (voire 8 192), `--batch-size 256`,
     `-ngl`/`--tensor-split` avec **2–3 Go de marge VRAM** ;
   - `Restart=on-failure` + `StartLimitBurst=3` (pas `Restart=always`).
3. **Sérialiser les charges GPU** : jamais `llama-server` + training en même
   temps. Utiliser un verrou (`flock /run/hera-gpu.lock`) ou
   `Conflicts=hera-training.service` dans les units systemd.
4. **Reboot après tout OOM/hang GPU**. Ne pas enchaîner un long run sur un
   driver qui vient d'OOM.
5. **Étendre `hera-observateur`** : GPU0 **et** GPU1, temp edge/junction/memory,
   power, VRAM utilisée/libre, clocks, throttle, CPU/RAM/IO ; conservation
   30 jours ; copie distante (DGX/NAS). Aujourd'hui il ne logge que `gpu0=0.0%`.
6. **Logs distants** : `netconsole` vers le DGX pour les messages noyau en
   temps réel ; journald persistant (déjà `Storage=persistent`).
7. **Vérifier kdump/pstore** : `kdump` configuré (`/var/crash/kexec_cmd`), mais
   pas testé. Tester avec un panic contrôlé **pendant une maintenance**.
8. **Alimentation/thermique** : vérifier le modèle/wattage du PSU (2×304 W GPU
   + CPU 170 W + périphériques ≈ 800–900 W ; marge transitoire nécessaire),
   les câbles PCIe (rails séparés), l'airflow, la courbe ventilo. Envisager un
   **onduleur**. Plafonner la puissance GPU (ex. 250 W) si le PSU est juste.
9. **Affichage** : si le freeze est Xorg/iGPU, passer l'affichage sur l'iGPU
   uniquement (blacklister les dGPU dans Xorg) ou passer le serveur en
   **headless** (SSH), pour qu'un hang dGPU ne fige pas le desktop.
10. **Vérifier les filesystems** après le REISUB : `dmesg` (ext4 errors),
    `fsck` en maintenance, `smartctl`.

### 5.2 Container pour les charges GPU

Spec recommandée (Podman ou Docker, image ROCm/PyTorch) :

```bash
podman run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --shm-size=16g \
  --cpus=8 --memory=64g --pids-limit=512 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -e HIP_VISIBLE_DEVICES=0,1 \
  -e PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8 \
  -v /home/jules/data:/data:ro \
  -v /home/jules/Documents/optizip/projects/bad-autoencoding:/work:ro \
  -v /home/jules/outputs:/outputs:rw \
  <image> bash
```

- **Limites CPU/RAM/IO** via cgroups : utiles et fiables.
- **La VRAM n'est pas limitable par cgroup** sur ROCm : ajouter dans le code
  `torch.cuda.set_per_process_memory_fraction(0.90)` et un **watchdog hôte**.
- **Pas de `--privileged`** ; monter les données en lecture seule ; sortie
  dans un volume dédié.
- Un **entrypoint préflight** vérifie : pas de `llama-server`, VRAM libre
  > 12 Go/GPU, temp < 80 °C, `dmesg` sans erreur, verrou GPU libre ; sinon il
  refuse de démarrer.

### 5.3 Garde-fou GPU hôte (`hera-gpu-guard.service`)

Un service systemd racine qui échantillonne toutes les 2–5 s :

- `rocm-smi --showtemp --showpower --showmeminfo vram --showuse` (GPU0/1/2) ;
- `dmesg`/journal pour `amdgpu` reset/hang ;
- log CSV + journald.

Seuils et actions (à ajuster) :

| Condition | Action |
|---|---|
| temp junction/memory > 95 °C | `systemctl stop` de l'unité GPU concernée (training ou llama) |
| puissance > 300 W/GPU (ou > plafond) | log + alerte ; stop si persistant |
| VRAM libre < 0,5 Go pendant > 30 s | stop de l'unité, pas du serveur |
| `amdgpu` reset/hang détecté | log, alerte, stop de l'unité |
| absence de heartbeat > 30 s | watchdog matériel/systemd (reboot contrôlé en dernier recours) |

Ne jamais tuer le serveur pour un problème applicatif : **tuer l'unité
fautive**, pas la machine. Le reboot watchdog est réservé au hang total.

Compléments :

- `hera-gpu-preflight` (script) appelé en `ExecStartPre` du training :
  vérifie verrou, VRAM, temp, absence de llama, `dmesg` propre.
- `systemd` : `RuntimeWatchdogSec=30s` si le watchdog matériel est disponible ;
  `Restart=on-failure` + rate limit pour les services.
- Le **training** dans un service `hera-training.service` (container) avec
  `Conflicts=llama-server.service` et `ExecStartPre=hera-gpu-preflight`.

### 5.4 Kernel / driver / matériel

- `amdgpu.lockup_timeout=10000` (ou 20000), `runpm=0`, `gpu_recovery=1`.
- Tester `pcie_aspm=off` et `iommu=pt` (un changement à la fois).
- Mettre à jour kernel/ROCm/firmware si un correctif connu existe ; sinon
  rester sur une version validée.
- Envisager un **reboot mensuel** (l'uptime de 44 jours est un facteur).
- `mcelog`/`edac-util`/`ras-mc-ctl`, `smartctl`, `memtest86+` si un doute
  matériel.
- Vérifier les erreurs PCIe/AER et les liens (`lspci -vv`).

### 5.5 Runbook de recovery après hard reboot

1. `dmesg -T` : erreurs ext4/xfs, amdgpu, PCIe.
2. `sudo fsck -f` sur les volumes ext4 si nécessaire (jamais sur un FS monté).
3. `sudo smartctl -a /dev/nvme0n1`.
4. Vérifier les services : nœud Hyperliquid, fathom, observateur ; llama
   **désactivé**.
5. Vérifier les backups et les données critiques.
6. Archiver `journalctl -b -1`, `kern.log`, `Xorg.0.log` dans
   `~/forensic-<date>/`.
7. Ne relancer un workload GPU qu'après le préflight et le guard.

**SysRq** : `kernel.sysrq=1` (ou masque 176 pour sync/remount/reboot) et
documenter la séquence sûre : `s` (sync), `u` (remount RO), `b` (reboot).
`r`/`k` étaient désactivés lors du crash ; les activer si utile.

### 5.6 Validation avant reprise

- Test de charge GPU 30 min (remplissage VRAM + compute) sous guard, avec
  monitoring temp/power/VRAM.
- Test du watchdog : tuer un process, vérifier que le guard agit.
- Test kdump/pstore pendant une maintenance.
- Test de reboot et de reprise des services.
- **Ne relancer le training qu'après** ces validations, dans le container,
  avec le préflight et le verrou.

---

## 6. Décisions / informations à obtenir de l'utilisateur

1. Déplacer `llama-server` sur le DGX (recommandé) ou le garder sur `hera`
   avec un contexte réduit ?
2. `hera` doit-il rester un desktop ou devenir **headless** (SSH) pour les
   calculs ?
3. Accepte-t-on un **reboot mensuel** de maintenance ?
4. Modèle/wattage du PSU, onduleur, câbles PCIe, températures boîtier.
5. Accès `sudo`/reboot pour appliquer GRUB, units systemd, kdump, netconsole.
6. Les services critiques (Hyperliquid, fathom) ont-ils des backups récents ?

---

## 7. Annexe — commandes déjà exécutées (résultats résumés)

- `journalctl --list-boots` : boot `-1` 2026-09-11 → 2026-10-24 00:50:12 ;
  boot `0` depuis 00:51:29.
- `journalctl -b -1 -k` : aucune erreur `amdgpu`/`kfd`/`mce`/`thermal`/`oom` ;
  dernières lignes = UFW + SysRq.
- `grep` crash window : observateur `gpu0=0.0%` jusqu'à 00:49:37, puis SysRq.
- `llama-server` boot -1 : arrêt propre à 11:39:45.
- `llama-server` boot 0 : démarré à 00:51:31, tué à 02:47:20, redémarré à
  02:47:20 (`Restart=always`), puis `stop`+`disable` par l'utilisateur.
- GPU maintenant : 15,82 / 15,85 GiB libres.
- `Xorg.0.log.old` : 2026-10-23 13:17 (pas le crash).
- `sar` : trou entre 00:07 et 00:50.
- `/var/crash/kexec_cmd` présent, pas de kdump/pstore.
- Cmdline : `amdgpu.lockup_timeout=1000` (suspect), double `crashkernel=`.

---

## 8. Première action de la conversation séparée

1. Lire ce fichier en entier.
2. Exécuter la section 4 et sauvegarder les sorties dans
   `~/forensic-20261024/`.
3. Présenter un **rapport forensic** : cause probable, causes exclues,
   preuves.
4. Proposer les quick wins 5.1 (GRUB, llama, observateur, netconsole) et
   attendre l'accord avant toute modification système ou reboot.
5. Ne pas lancer de training GPU avant la validation 5.6.
6. Le run vision fidèle reste sur le **DGX** (ne pas toucher au conv en
   cours) ; `hera` ne doit pas être utilisé pour un run long tant que le
   garde-fou n'est pas validé.
