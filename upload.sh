#!/bin/bash
# GitHub Upload Script for Virtual Earth Language Evolution

set -e

echo "🚀 Uploading Virtual Earth project to GitHub..."

# Your GitHub credentials
GITHUB_USERNAME="MauveAndromeda"
GITHUB_TOKEN="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" 
REPO_NAME="virtual-earth-language"

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -f "README.md" ]; then
    echo "❌ Error: Run this script from your virtual-earth-language directory"
    exit 1
fi

# Create repository on GitHub using API
echo "📦 Creating repository on GitHub..."
curl -u "$GITHUB_USERNAME:$GITHUB_TOKEN" \
     -X POST \
     -H "Accept: application/vnd.github.v3+json" \
     https://api.github.com/user/repos \
     -d "{
         \"name\":\"$REPO_NAME\",
         \"description\":\"🌍 Virtual Earth: Emergent Language Evolution in Multi-Agent Systems - Ubuntu optimized with CUDA support\",
         \"private\":false,
         \"auto_init\":false
     }"

# Wait a moment for GitHub to create the repo
echo "⏳ Waiting for repository creation..."
sleep 2

# Check if remote already exists and remove it
if git remote get-url origin >/dev/null 2>&1; then
    echo "🔄 Removing existing remote..."
    git remote remove origin
fi

# Add remote with authentication
echo "🔗 Adding GitHub remote..."
git remote add origin "https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git"

# Set up proper branch
echo "🌿 Setting up main branch..."
if git branch --show-current | grep -q "master"; then
    git branch -M main
fi

# Create README badge and final commit
echo "📝 Adding project badges..."
cat > .github-badges.md << 'EOF'
# Badges for README
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange.svg)](https://ubuntu.com/)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
EOF

# Add everything and commit
git add -A
git status
read -p "📋 Review the files above. Continue with upload? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Upload cancelled"
    exit 1
fi

# Final commit
git commit -m "🚀 Upload complete Virtual Earth Language Evolution project

✨ Features:
- Ubuntu 24.04 optimized setup
- Multi-agent emergent language framework  
- Geographic constraints and population dynamics
- Multi-objective optimization (success + MI + topology)
- Real-time visualization system
- Complete academic research structure
- CPU + GPU support with auto-detection

🧪 Tested and working:
- PyTorch 2.4.0 with CPU support
- Conda environment with all dependencies
- Experiment pipeline functional
- 768-dimensional observation space
- Hydra configuration management"

# Push to GitHub
echo "⬆️ Pushing to GitHub..."
git push -u origin main

# Clean up credentials from git config
echo "🧹 Cleaning up..."
git remote set-url origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"

echo "✅ Success! Your project is now at:"
echo "🌐 https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo ""
echo "🔒 Security note: Remember to revoke tokens when done!"

# Optional: Open in browser
if command -v xdg-open >/dev/null 2>&1; then
    read -p "🌐 Open repository in browser? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open "https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    fi
fi

echo "🎉 Upload complete! Happy researching!"
