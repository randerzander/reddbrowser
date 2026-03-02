package com.reddbrowser.android.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.reddbrowser.android.data.ApiKeyStore
import com.reddbrowser.android.data.AppRepository
import com.reddbrowser.android.data.CommentNode
import com.reddbrowser.android.data.FeedSource
import com.reddbrowser.android.data.PostDetail
import com.reddbrowser.android.data.PostItem
import com.reddbrowser.android.data.PythonBridge
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class FeedUiState(
    val source: FeedSource = FeedSource.REDDIT,
    val subreddit: String = "localllama",
    val subreddits: List<String> = listOf("localllama", "news.ycombinator.com", "vibecoding", "linux", "ExperiencedDevs"),
    val loading: Boolean = false,
    val error: String? = null,
    val posts: List<PostItem> = emptyList()
)

data class DetailUiState(
    val loading: Boolean = false,
    val error: String? = null,
    val post: PostItem? = null,
    val comments: List<CommentNode> = emptyList(),
    val aiText: String = "",
    val aiLoading: Boolean = false,
    val aiPrompt: String = ""
)

data class SettingsUiState(
    val apiKey: String = "",
    val baseModel: String = ApiKeyStore.DEFAULT_BASE_MODEL,
    val vlmModel: String = ApiKeyStore.DEFAULT_VLM_MODEL
)

class AppViewModel(application: Application) : AndroidViewModel(application) {
    private val keyStore = ApiKeyStore(application)
    private val repository = AppRepository(PythonBridge(application))

    private val _feedState = MutableStateFlow(FeedUiState())
    val feedState: StateFlow<FeedUiState> = _feedState.asStateFlow()

    private val _detailState = MutableStateFlow(DetailUiState())
    val detailState: StateFlow<DetailUiState> = _detailState.asStateFlow()

    private val _settingsState = MutableStateFlow(
        SettingsUiState(
            apiKey = keyStore.getApiKey(),
            baseModel = keyStore.getBaseModel(),
            vlmModel = keyStore.getVlmModel()
        )
    )
    val settingsState: StateFlow<SettingsUiState> = _settingsState.asStateFlow()

    init {
        loadFeed()
    }

    fun updateSource(source: FeedSource) {
        _feedState.update {
            val nextSubreddit = if (source == FeedSource.HN) "news.ycombinator.com" else it.subreddit
            it.copy(source = source, subreddit = nextSubreddit)
        }
    }

    fun updateSubreddit(subreddit: String) {
        _feedState.update { it.copy(subreddit = subreddit) }
    }

    fun loadFeed() {
        val snapshot = _feedState.value
        viewModelScope.launch {
            _feedState.update { it.copy(loading = true, error = null) }
            runCatching {
                repository.loadFeed(snapshot.source, snapshot.subreddit)
            }.onSuccess { result ->
                _feedState.update { it.copy(loading = false, posts = result.posts) }
            }.onFailure { error ->
                _feedState.update { it.copy(loading = false, error = error.message ?: "Failed to load feed") }
            }
        }
    }

    fun openPost(post: PostItem) {
        viewModelScope.launch {
            _detailState.update {
                it.copy(
                    loading = true,
                    error = null,
                    post = post,
                    comments = emptyList(),
                    aiText = ""
                )
            }
            runCatching { repository.loadPostDetail(post) }
                .onSuccess { detail ->
                    _detailState.update {
                        it.copy(
                            loading = false,
                            post = detail.post,
                            comments = detail.commentsTree
                        )
                    }
                }
                .onFailure { error ->
                    _detailState.update {
                        it.copy(loading = false, error = error.message ?: "Failed to load post detail")
                    }
                }
        }
    }

    fun closeDetail() {
        _detailState.value = DetailUiState()
    }

    fun updateAiPrompt(value: String) {
        _detailState.update { it.copy(aiPrompt = value) }
    }

    fun runTextSummary() {
        val post = _detailState.value.post ?: return
        val text = post.data.selftext
        if (text.isBlank()) {
            _detailState.update { it.copy(aiText = "No post text available for summary.") }
            return
        }
        runAiTask {
            repository.summarizeText(text, keyStore.getApiKey(), keyStore.getBaseModel()).summary
        }
    }

    fun runArticleSummary() {
        val post = _detailState.value.post ?: return
        if (post.data.url.isBlank()) {
            _detailState.update { it.copy(aiText = "No URL available for article summarization.") }
            return
        }
        runAiTask {
            repository.summarizeArticle(post.data.url, keyStore.getApiKey(), keyStore.getBaseModel()).summary
        }
    }

    fun runCommentSummary() {
        val comments = _detailState.value.comments
        runAiTask {
            val text = repository.topCommentsText(comments)
            repository.summarizeComments(text, keyStore.getApiKey(), keyStore.getBaseModel()).summary
        }
    }

    fun runAskAi() {
        val prompt = _detailState.value.aiPrompt.trim()
        if (prompt.isBlank()) {
            _detailState.update { it.copy(aiText = "Enter a prompt first.") }
            return
        }
        val post = _detailState.value.post ?: return
        val comments = _detailState.value.comments
        val currentSummary = _detailState.value.aiText
        runAiTask {
            val context = repository.buildAiContext(post, comments, currentSummary)
            repository.askAi(context, prompt, keyStore.getApiKey(), keyStore.getBaseModel()).response
        }
    }

    fun updateSettings(apiKey: String, baseModel: String, vlmModel: String) {
        keyStore.setApiKey(apiKey)
        keyStore.setBaseModel(baseModel)
        keyStore.setVlmModel(vlmModel)
        _settingsState.value = SettingsUiState(apiKey = apiKey, baseModel = baseModel, vlmModel = vlmModel)
    }

    private fun runAiTask(task: suspend () -> String) {
        if (keyStore.getApiKey().isBlank()) {
            _detailState.update { it.copy(aiText = "Missing OpenRouter API key. Set it in Settings.") }
            return
        }
        viewModelScope.launch {
            _detailState.update { it.copy(aiLoading = true, error = null) }
            runCatching { task() }
                .onSuccess { output ->
                    _detailState.update { it.copy(aiLoading = false, aiText = output) }
                }
                .onFailure { error ->
                    _detailState.update { it.copy(aiLoading = false, aiText = "AI error: ${error.message}") }
                }
        }
    }
}
