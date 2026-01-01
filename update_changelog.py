#!/usr/bin/env python3
import os
import subprocess
from datetime import datetime
import argparse

# Configuration
CHANGELOG_FILE = "CHANGELOG.md"
CODE_FILE = "dataset_forge.py"
MARKER = "## 📋 Version History"

USAGE_GUIDE = """
MODE D'UTILISATION IDÉAL (WORKFLOW):

1. Effectuez vos modifications (code, features, fix...).
2. Mettez à jour la version dans le code (dataset_forge.py).
3. Générez le Changelog :
   python update_changelog.py
   (Le script détecte automatiquement la version dans dataset_forge.py).
4. Commitez le tout :
   git add .
   git commit -m "chore: release v1.5.4"
5. Créez le Tag :
   git tag -a v1.5.4 -m "Version 1.5.4"
6. Envoyez :
   git push && git push --tags

POUR GÉNÉRER UN HISTORIQUE PASSÉ :
   python update_changelog.py --range v1.0..v1.1 --version "[v1.1.0]"
"""

def get_version_from_code():
    """Extrait la version depuis le fichier python source."""
    if not os.path.exists(CODE_FILE):
        return None
    
    with open(CODE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith("__version__"):
                # Format attendu: __version__ = "1.5.4"
                try:
                    return line.split('=')[1].strip().strip('"').strip("'")
                except IndexError:
                    pass
    return None

def check_uncommitted_changes():
    """Vérifie s'il y a des changements non commités."""
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, encoding='utf-8')
        if result.stdout.strip():
            print("\n⚠️  ATTENTION : Des modifications non commités ont été détectées !")
            print("   Le script se base uniquement sur l'historique GIT (les commits).")
            print("   Vos fichiers modifiés actuels ne seront pas listés tant qu'ils ne sont pas commités.")
            print("   Conseil : Faites vos commits (ex: 'feat: ...') AVANT de lancer ce script.\n")
    except Exception:
        pass

def get_git_commits(revision_range=None):
    """Récupère les commits depuis le dernier tag ou tous les commits si aucun tag."""
    if not revision_range:
        try:
            # Trouve le dernier tag
            tag = subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"], stderr=subprocess.DEVNULL).decode().strip()
            revision_range = f"{tag}..HEAD"
            print(f"ℹ️  Dernier tag détecté : {tag}")
        except subprocess.CalledProcessError:
            # Pas de tag trouvé, on prend tout
            revision_range = "HEAD"
            print("ℹ️  Aucun tag trouvé, analyse de tout l'historique.")
    
    print(f"ℹ️  Plage de révision analysée : {revision_range}")

    # Récupère les messages de commit
    result = subprocess.run(["git", "log", revision_range, "--pretty=format:%s"], capture_output=True, text=True, encoding='utf-8')
    commits = result.stdout.strip().split('\n')
    return [c for c in commits if c] # Filtre les lignes vides

def format_commits(commits, version_name=None):
    """Trie les commits par catégories basées sur des préfixes courants."""
    categories = {
        "✨ New Features": ["feat", "add", "new", "ajout"],
        "🐛 Bug Fixes": ["fix", "bug", "patch", "corr"],
        "🔧 Maintenance & Improvements": ["chore", "refactor", "docs", "style", "perf", "update"]
    }
    
    grouped = {k: [] for k in categories}
    others = []
    
    for commit in commits:
        if not commit: continue
        matched = False
        lower_commit = commit.lower()
        
        for cat, keywords in categories.items():
            if any(lower_commit.startswith(kw) for kw in keywords):
                # On garde le message tel quel
                grouped[cat].append(commit)
                matched = True
                break
        
        if not matched:
            others.append(commit)
            
    # Construction du Markdown
    lines = []
    date_str = datetime.now().strftime("%Y-%m-%d")
    header = version_name if version_name else "[Unreleased]"
    lines.append(f"### {header} - {date_str}")
    
    for cat, msgs in grouped.items():
        if msgs:
            lines.append(f"\n#### {cat}")
            for msg in msgs:
                lines.append(f"- {msg}")
                
    if others:
        lines.append("\n#### 🔧 Others")
        for msg in others:
            lines.append(f"- {msg}")
            
    return "\n".join(lines)

def update_file(new_content):
    if not os.path.exists(CHANGELOG_FILE):
        print(f"Erreur: {CHANGELOG_FILE} n'existe pas.")
        return

    with open(CHANGELOG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    if MARKER not in content:
        print(f"Erreur: Marqueur '{MARKER}' introuvable dans le fichier.")
        return

    parts = content.split(MARKER)
    # Insertion du nouveau bloc juste après le marqueur
    new_full_content = parts[0] + MARKER + "\n\n" + new_content + "\n" + parts[1]
    
    with open(CHANGELOG_FILE, 'w', encoding='utf-8') as f:
        f.write(new_full_content)
    print(f"✅ {CHANGELOG_FILE} mis à jour avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mise à jour automatique du CHANGELOG.md à partir de l'historique git.",
        epilog=USAGE_GUIDE,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--range", help="Git revision range (e.g. v1.0..v1.1)")
    parser.add_argument("--version", help="Version name (e.g. [v1.1.0])")
    args = parser.parse_args()

    version_name = args.version
    if not version_name:
        detected_ver = get_version_from_code()
        if detected_ver:
            version_name = f"[v{detected_ver}]"
            print(f"ℹ️  Version détectée automatiquement : {version_name}")

    check_uncommitted_changes()
    commits = get_git_commits(args.range)
    
    if not commits:
        print("Aucun nouveau commit à ajouter.")
    else:
        formatted = format_commits(commits, version_name)
        update_file(formatted)