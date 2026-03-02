package com.reddbrowser.android.data

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class PythonBridge(context: Context) {
    private val module: PyObject

    init {
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(context))
        }
        module = Python.getInstance().getModule("reddbrowser_bridge.bridge")
    }

    fun listPosts(source: String, subreddit: String, limit: Int, pageToken: String?): String =
        module.callAttr("list_posts", source, subreddit, limit, pageToken ?: "").toString()

    fun getPostDetail(source: String, postIdOrPermalink: String): String =
        module.callAttr("get_post_detail", source, postIdOrPermalink).toString()

    fun summarizeText(text: String, apiKey: String, model: String): String =
        module.callAttr("summarize_text", text, apiKey, model).toString()

    fun summarizeArticle(url: String, apiKey: String, model: String): String =
        module.callAttr("summarize_article", url, apiKey, model).toString()

    fun summarizeComments(commentsText: String, apiKey: String, model: String): String =
        module.callAttr("summarize_comments", commentsText, apiKey, model).toString()

    fun askAi(context: String, prompt: String, apiKey: String, model: String): String =
        module.callAttr("ask_ai", context, prompt, apiKey, model).toString()
}
