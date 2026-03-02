# ReddBrowser Android (Chaquopy + Compose)

This directory contains the Android implementation scaffold for ReddBrowser.

## What is implemented
- Native Android app shell with Jetpack Compose UI.
- Embedded Python bridge (`app/src/main/python/reddbrowser_bridge`) for:
  - Reddit feed + post detail + comments tree
  - Hacker News feed + post detail + comments tree
  - OpenRouter-backed text/article/comment summaries
  - OpenRouter-backed Q&A (`ask_ai`)
- Encrypted on-device settings storage for:
  - OpenRouter API key
  - Base model
  - VLM model

## Build prerequisites
- Android Studio Iguana+ (or equivalent AGP/Kotlin support)
- Android SDK 34
- JDK 17

## Build/run
1. Open `android/` as the Android Studio project root.
2. Let Gradle sync and install dependencies.
3. Run on an emulator/device (minSdk 26).

## Notes
- This v1 Android code intentionally excludes Twitter/Twikit flows.
- API key is user-provided in Settings and stored via `EncryptedSharedPreferences`.
