**What this changes**

**What certifies it** (the test, and its oracle)

- [ ] `./gradlew test` passes locally
- [ ] The new test fails with the fix reverted
- [ ] Public API changes come with an updated `api/*.api` (`./gradlew updateKotlinAbi`)
- [ ] Unsupported inputs refuse with an error that names them
- [ ] No edits under `third-party/` outside the procedure in `docs/vendoring.md`
