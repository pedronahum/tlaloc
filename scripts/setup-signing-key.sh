#!/usr/bin/env bash
# §0.4.510 — generate the PGP signing key Maven Central requires, and wire it into
# Gradle, without any secret reaching a terminal transcript or a tracked file.
#
# RUN THIS IN YOUR OWN TERMINAL, not through an agent or a CI job: it prompts for a
# passphrase on a TTY, and the passphrase is the one thing here that must not be
# pasted anywhere it could be recorded.
#
#   bash scripts/setup-signing-key.sh
#
# What it does, in order:
#   1. generates an RSA-4096 signing key (2-year expiry — renewable; an expired key
#      invalidates nothing already published, because Central records the signature
#      at publish time)
#   2. publishes the PUBLIC half to two keyservers, because Central verifies the
#      signature by looking the key up
#   3. exports the PRIVATE half armoured, folds it to one line, and writes it plus
#      the passphrase into ~/.gradle/gradle.properties as `signingInMemoryKey` /
#      `signingInMemoryKeyPassword` (the names the root build.gradle.kts reads)
#
# It prints the key id and fingerprint. It never prints the key or the passphrase.
#
# On storing the passphrase in a file at all: that is what an unattended signer
# requires, and `~/.gradle/gradle.properties` is chmod 600 and outside every git
# repository. This repository's OWN gradle.properties is tracked — a secret there
# would be committed and pushed, which is why nothing here writes to it.
set -euo pipefail

PROPS="$HOME/.gradle/gradle.properties"
NAME="${SIGNING_NAME:-$(git config user.name  || true)}"
EMAIL="${SIGNING_EMAIL:-$(git config user.email || true)}"

die() { printf '\nerror: %s\n' "$1" >&2; exit 1; }

[ -t 0 ] || die "no TTY. Run this directly in a terminal — it must prompt for a passphrase."
command -v gpg >/dev/null || die "gpg not found."
[ -n "$NAME" ]  || die "no name. Set git config user.name, or export SIGNING_NAME."
[ -n "$EMAIL" ] || die "no email. Set git config user.email, or export SIGNING_EMAIL."

printf 'Signing key identity:\n  %s <%s>\n' "$NAME" "$EMAIL"
printf 'Override with SIGNING_NAME / SIGNING_EMAIL if that is wrong.\n\n'

if gpg --list-secret-keys "$EMAIL" >/dev/null 2>&1; then
    die "a secret key for <$EMAIL> already exists. Delete it first (gpg --delete-secret-and-public-key <id>), or export SIGNING_EMAIL to use a different identity. Refusing to create a second key for the same address, because then it is ambiguous which one signed a release."
fi

# Read the passphrase twice. -s so it is not echoed; never logged, never argv.
read -rsp 'Passphrase for the new key: ' PASS; echo
[ ${#PASS} -ge 12 ] || die "use at least 12 characters — this key signs artifacts other people install."
read -rsp 'Again: ' PASS2; echo
[ "$PASS" = "$PASS2" ] || die "the two passphrases differ."
unset PASS2
echo

# The batch file carries the passphrase, so: private temp dir, removed on any exit.
umask 077
WORK="$(mktemp -d)"
cleanup() {
    if [ -d "$WORK" ]; then
        find "$WORK" -type f -exec shred -u {} + 2>/dev/null || true
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT INT TERM

cat > "$WORK/params" <<EOF
%echo Generating an RSA-4096 signing key…
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: $NAME
Name-Email: $EMAIL
Expire-Date: 2y
Passphrase: $PASS
%commit
%echo done
EOF

gpg --batch --generate-key "$WORK/params"
find "$WORK" -name params -exec shred -u {} + 2>/dev/null || true

KEYID="$(gpg --list-secret-keys --with-colons "$EMAIL" | awk -F: '/^fpr:/ {print $10; exit}')"
[ -n "$KEYID" ] || die "the key was generated but its fingerprint could not be read."
printf '\nkey id: %s\n' "$KEYID"

# 2. The public half, to the servers Central looks a signature up on. keys.openpgp.org
#    withholds the identity (name/email) until you confirm the address by email; the
#    key material itself is served either way, which is what signature verification
#    needs. Failures here are not fatal — the key exists locally and can be pushed
#    again — so they are reported, not raised.
for KS in hkps://keys.openpgp.org hkps://keyserver.ubuntu.com; do
    printf 'publishing public key to %s … ' "$KS"
    if gpg --keyserver "$KS" --send-keys "$KEYID" >/dev/null 2>&1; then
        echo ok
    else
        echo "FAILED (retry later: gpg --keyserver $KS --send-keys $KEYID)"
    fi
done

# 3. Wire Gradle. The armoured key folded to one line with literal \n, which is what
#    useInMemoryPgpKeys expects from a properties file.
ARMOURED="$(gpg --batch --yes --pinentry-mode loopback --passphrase "$PASS" \
                --armor --export-secret-keys "$KEYID" | sed -z 's/\n/\\n/g')"
[ -n "$ARMOURED" ] || die "exporting the secret key produced nothing."

mkdir -p "$(dirname "$PROPS")"
touch "$PROPS"
chmod 600 "$PROPS"
# Drop any previous signing lines (commented placeholders included) and append fresh.
TMP_PROPS="$WORK/props"
grep -v -E '^#?signingInMemoryKey(Password)?=' "$PROPS" > "$TMP_PROPS" || true
{
    cat "$TMP_PROPS"
    printf 'signingInMemoryKey=%s\n' "$ARMOURED"
    printf 'signingInMemoryKeyPassword=%s\n' "$PASS"
} > "$PROPS"
chmod 600 "$PROPS"
unset PASS ARMOURED

cat <<EOF

Wired into $PROPS (chmod 600, outside every git repository).

Next, and neither step sends anything to Central:
  ./gradlew publishToMavenLocal          # signing now runs; .asc files appear in ~/.m2
  ./gradlew publishAllPublicationsToCentralRepository --dry-run

Key id for the Central Portal, and for renewing in two years:
  $KEYID
EOF
