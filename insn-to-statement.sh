set -x
set -e
sed -i s/Statement/Statement/g $(git ls-files | grep -v compyte)
sed -i s/statement/statement/g $(git ls-files | grep -v compyte)
sed -i s/STATEMENT/STATEMENT/g $(git ls-files | grep -v compyte)
sed -i s/stmt/stmt/g $(git ls-files | grep -v compyte)
for d in kernel codegen transform; do
  git mv loopy/$d/statement.py loopy/$d/statement.py
done
patch -p1 < ./stmt-compat-fixes.patch
