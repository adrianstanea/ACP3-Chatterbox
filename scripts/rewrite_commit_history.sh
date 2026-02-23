#!/bin/bash
# Script to rewrite commit history:
# - Change all commit authors to current git user
# - Change all commit committers to current git user
# - Remove all Co-Authored-By lines
# - Remove existing Signed-off-by lines
# - Re-sign commits with current git user

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Git History Rewrite Tool ===${NC}"
echo ""

# Get current git user info
CURRENT_USER=$(git config user.name)
CURRENT_EMAIL=$(git config user.email)

if [ -z "$CURRENT_USER" ] || [ -z "$CURRENT_EMAIL" ]; then
    echo -e "${RED}Error: Git user.name and user.email must be configured${NC}"
    echo "Run: git config user.name 'Your Name'"
    echo "     git config user.email 'your.email@example.com'"
    exit 1
fi

echo -e "Current git user: ${GREEN}$CURRENT_USER <$CURRENT_EMAIL>${NC}"
echo ""

# Get the branch to rewrite
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "Current branch: ${GREEN}$CURRENT_BRANCH${NC}"
echo ""

# Ask for base commit (where to start rewriting from)
echo "Enter the base commit to rewrite FROM (or press Enter for ALL commits):"
echo "  - Use commit hash (e.g., abc1234)"
echo "  - Use branch name (e.g., main)"
echo "  - Or leave empty to rewrite ALL commits"
read -r BASE_COMMIT

if [ -z "$BASE_COMMIT" ]; then
    # Rewrite all commits on current branch
    RANGE_ARG="HEAD"
    BASE_COMMIT_DISPLAY="all commits"
    echo -e "${YELLOW}Rewriting ALL commits${NC}"
else
    # Validate commit exists
    if ! git rev-parse "$BASE_COMMIT" >/dev/null 2>&1; then
        echo -e "${RED}Error: Commit '$BASE_COMMIT' not found${NC}"
        exit 1
    fi
    RANGE_ARG="$BASE_COMMIT..HEAD"
    BASE_COMMIT_DISPLAY="commits after $BASE_COMMIT"
    echo -e "${YELLOW}Rewriting commits after: $BASE_COMMIT${NC}"
fi
echo ""

# Warning
echo -e "${RED}WARNING: This will rewrite git history!${NC}"
echo -e "${RED}This requires a force push and will affect all collaborators.${NC}"
echo ""
echo "Changes that will be made to ${BASE_COMMIT_DISPLAY}:"
echo "  ✗ Change all authors to: $CURRENT_USER <$CURRENT_EMAIL>"
echo "  ✗ Change all committers to: $CURRENT_USER <$CURRENT_EMAIL>"
echo "  ✗ Remove all 'Co-Authored-By:' lines"
echo "  ✗ Remove all existing 'Signed-off-by:' lines"
echo "  ✓ Add new 'Signed-off-by: $CURRENT_USER <$CURRENT_EMAIL>'"
echo ""
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${YELLOW}Creating backup branch...${NC}"
BACKUP_BRANCH="${CURRENT_BRANCH}-backup-$(date +%Y%m%d-%H%M%S)"
git branch "$BACKUP_BRANCH"
echo -e "${GREEN}✓ Backup created: $BACKUP_BRANCH${NC}"
echo ""

echo -e "${YELLOW}Rewriting commit history...${NC}"

# Create a temporary script for the message filter
FILTER_SCRIPT=$(mktemp)
cat > "$FILTER_SCRIPT" << 'FILTER_EOF'
#!/bin/bash
# Read the entire commit message
msg=$(cat)

# Remove Co-Authored-By and Signed-off-by lines, preserving everything else
filtered_msg=$(echo "$msg" | sed '/^Co-Authored-By:/d' | sed '/^Signed-off-by:/d')

# Remove trailing blank lines
filtered_msg=$(echo "$filtered_msg" | sed -e :a -e '/^\s*$/d;N;ba')

# Output the filtered message with new sign-off
echo "$filtered_msg"
echo ""
echo "Signed-off-by: $NEW_GIT_USER <$NEW_GIT_EMAIL>"
FILTER_EOF

chmod +x "$FILTER_SCRIPT"

# Export new git user info
export NEW_GIT_USER="$CURRENT_USER"
export NEW_GIT_EMAIL="$CURRENT_EMAIL"

# Clean up any previous filter-branch refs
rm -rf .git/refs/original/

# Use git filter-branch to rewrite author, committer, and commit messages
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
    --env-filter '
        export GIT_AUTHOR_NAME="'"$CURRENT_USER"'"
        export GIT_AUTHOR_EMAIL="'"$CURRENT_EMAIL"'"
        export GIT_COMMITTER_NAME="'"$CURRENT_USER"'"
        export GIT_COMMITTER_EMAIL="'"$CURRENT_EMAIL"'"
    ' \
    --msg-filter "$FILTER_SCRIPT" \
    -- "$RANGE_ARG"

# Clean up
rm "$FILTER_SCRIPT"
rm -rf .git/refs/original/

echo ""
echo -e "${GREEN}✓ History rewritten successfully!${NC}"
echo ""

# Show comparison
echo -e "${YELLOW}=== Commit comparison ===${NC}"
echo ""
echo -e "${YELLOW}Before (backup):${NC}"
git log --oneline "$BACKUP_BRANCH" -5
echo ""
echo -e "${YELLOW}After (current):${NC}"
git log --oneline HEAD -5
echo ""

# Show a sample commit with full details
echo -e "${YELLOW}=== Sample rewritten commit (latest) ===${NC}"
git log -1 --pretty=format:"Author: %an <%ae>%nCommitter: %cn <%ce>%nDate: %ad%n%nCommit Message:%n%B" HEAD
echo ""
echo ""

echo -e "${GREEN}✓ Complete!${NC}"
echo ""
echo "Verify the changes:"
echo "  git log --pretty=format:\"%h - %an <%ae>%n%s%n%b%n---\" -3"
echo ""
echo "Next steps:"
echo "  1. Review more commits: git log -5"
echo "  2. If satisfied, force push: ${YELLOW}git push origin $CURRENT_BRANCH --force${NC}"
echo "  3. If not satisfied, restore: ${YELLOW}git reset --hard $BACKUP_BRANCH${NC}"
echo ""
echo "Backup branch: $BACKUP_BRANCH"
echo -e "${RED}Remember: Force push will rewrite history for all collaborators!${NC}"
