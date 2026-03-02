package com.reddbrowser.android.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class BridgeError(
    val code: String,
    val message: String,
    val retryable: Boolean = false
)

@Serializable
data class PostData(
    val id: String = "",
    val title: String = "",
    val author: String = "",
    val score: Int = 0,
    @SerialName("num_comments") val numComments: Int = 0,
    val url: String = "",
    val permalink: String = "",
    val selftext: String = "",
    @SerialName("created_at") val createdAt: String? = null,
    @SerialName("hn_comments_url") val hnCommentsUrl: String? = null,
    @SerialName("hn_id") val hnId: String? = null
)

@Serializable
data class PostItem(
    val source: String,
    val data: PostData
)

@Serializable
data class FeedResult(
    val posts: List<PostItem>,
    @SerialName("next_page_token") val nextPageToken: String? = null
)

@Serializable
data class CommentData(
    val id: String,
    val author: String,
    val body: String,
    val score: Int = 0
)

@Serializable
data class CommentNode(
    val data: CommentData,
    val replies: List<CommentNode> = emptyList(),
    val level: Int = 0
)

@Serializable
data class PostDetail(
    val post: PostItem,
    @SerialName("comments_tree") val commentsTree: List<CommentNode>
)

@Serializable
data class SummaryResult(
    val summary: String
)

@Serializable
data class AskResult(
    val response: String
)

enum class FeedSource(val wireValue: String) {
    REDDIT("reddit"),
    HN("hn")
}
