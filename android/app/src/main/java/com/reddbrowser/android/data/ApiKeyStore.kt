package com.reddbrowser.android.data

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

class ApiKeyStore(context: Context) {
    private val prefs: SharedPreferences = run {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context,
            "reddbrowser_secure",
            masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
        )
    }

    fun getApiKey(): String = prefs.getString(KEY_OPENROUTER, "") ?: ""

    fun setApiKey(value: String) {
        prefs.edit().putString(KEY_OPENROUTER, value.trim()).apply()
    }

    fun getBaseModel(): String = prefs.getString(KEY_BASE_MODEL, DEFAULT_BASE_MODEL) ?: DEFAULT_BASE_MODEL

    fun setBaseModel(value: String) {
        prefs.edit().putString(KEY_BASE_MODEL, value.trim()).apply()
    }

    fun getVlmModel(): String = prefs.getString(KEY_VLM_MODEL, DEFAULT_VLM_MODEL) ?: DEFAULT_VLM_MODEL

    fun setVlmModel(value: String) {
        prefs.edit().putString(KEY_VLM_MODEL, value.trim()).apply()
    }

    companion object {
        private const val KEY_OPENROUTER = "openrouter_api_key"
        private const val KEY_BASE_MODEL = "base_llm_model"
        private const val KEY_VLM_MODEL = "vlm_model"

        const val DEFAULT_BASE_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
        const val DEFAULT_VLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
    }
}
