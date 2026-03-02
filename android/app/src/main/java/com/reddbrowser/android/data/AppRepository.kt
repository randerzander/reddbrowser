package com.reddbrowser.android.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.decodeFromJsonElement
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class AppRepository(
    private val bridge: PythonBridge
) {
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun loadFeed(source: FeedSource, subreddit: String, limit: Int = 50): FeedResult = withContext(Dispatchers.IO) {
        val raw = bridge.listPosts(source.wireValue, subreddit, limit, null)
        parseResult(raw)
    }

    suspend fun loadPostDetail(post: PostItem): PostDetail = withContext(Dispatchers.IO) {
        val idOrPermalink = if (post.source == "hn") post.data.id else post.data.permalink
        val raw = bridge.getPostDetail(post.source, idOrPermalink)
        parseResult(raw)
    }

    suspend fun summarizeText(text: String, apiKey: String, model: String): SummaryResult = withContext(Dispatchers.IO) {
        parseResult(bridge.summarizeText(text, apiKey, model))
    }

    suspend fun summarizeArticle(url: String, apiKey: String, model: String): SummaryResult = withContext(Dispatchers.IO) {
        parseResult(bridge.summarizeArticle(url, apiKey, model))
    }

    suspend fun summarizeComments(commentsText: String, apiKey: String, model: String): SummaryResult = withContext(Dispatchers.IO) {
        parseResult(bridge.summarizeComments(commentsText, apiKey, model))
    }

    suspend fun askAi(context: String, prompt: String, apiKey: String, model: String): AskResult = withContext(Dispatchers.IO) {
        parseResult(bridge.askAi(context, prompt, apiKey, model))
    }

    fun topCommentsText(comments: List<CommentNode>, limit: Int = 10): String {
        if (comments.isEmpty()) return "No comments available."
        return comments.take(limit).mapIndexed { index, node ->
            "${index + 1}. Author: ${node.data.author}, Score: ${node.data.score}\n   Comment: ${node.data.body}"
        }.joinToString("\n")
    }

    fun buildAiContext(post: PostItem, comments: List<CommentNode>, currentSummary: String): String {
        val postText = post.data.selftext.ifBlank { "No post text provided." }
        return buildString {
            appendLine("Context about the post:")
            appendLine("- Post text: $postText")
            appendLine()
            appendLine("AI-generated content:")
            appendLine(currentSummary.ifBlank { "No summary generated yet." })
            appendLine()
            appendLine("Top comments:")
            appendLine(topCommentsText(comments))
        }
    }

    private inline fun <reified T> parseResult(raw: String): T {
        val root = json.parseToJsonElement(raw).jsonObject
        val ok = root["ok"]?.jsonPrimitive?.content?.toBooleanStrictOrNull() ?: false
        if (!ok) {
            val error = root["error"]?.jsonObject
            val message = error?.get("message")?.jsonPrimitive?.content ?: "Unknown error"
            throw IllegalStateException(message)
        }
        val result = root["result"] ?: throw IllegalStateException("Missing result payload")
        return json.decodeFromJsonElement(result)
    }
}
