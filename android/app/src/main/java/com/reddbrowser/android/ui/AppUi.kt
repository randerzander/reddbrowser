package com.reddbrowser.android.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.reddbrowser.android.data.CommentNode
import com.reddbrowser.android.data.FeedSource
import com.reddbrowser.android.data.PostItem

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReddBrowserApp(viewModel: AppViewModel) {
    val feedState by viewModel.feedState.collectAsStateWithLifecycle()
    val detailState by viewModel.detailState.collectAsStateWithLifecycle()
    val settingsState by viewModel.settingsState.collectAsStateWithLifecycle()
    var showSettings by rememberSaveable { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(if (detailState.post == null) "ReddBrowser Android" else detailState.post?.data?.title ?: "")
                },
                navigationIcon = {
                    if (detailState.post != null) {
                        IconButton(onClick = viewModel::closeDetail) {
                            Icon(Icons.Default.ArrowBack, contentDescription = "Back")
                        }
                    }
                },
                actions = {
                    IconButton(onClick = { showSettings = true }) {
                        Icon(Icons.Default.Settings, contentDescription = "Settings")
                    }
                }
            )
        }
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            if (detailState.post == null) {
                FeedScreen(
                    state = feedState,
                    onSourceChange = viewModel::updateSource,
                    onSubredditChange = viewModel::updateSubreddit,
                    onRefresh = viewModel::loadFeed,
                    onOpenPost = viewModel::openPost
                )
            } else {
                DetailScreen(
                    state = detailState,
                    onSummaryText = viewModel::runTextSummary,
                    onSummaryComments = viewModel::runCommentSummary,
                    onSummaryArticle = viewModel::runArticleSummary,
                    onPromptChange = viewModel::updateAiPrompt,
                    onAskAi = viewModel::runAskAi
                )
            }
        }
    }

    if (showSettings) {
        SettingsDialog(
            state = settingsState,
            onDismiss = { showSettings = false },
            onSave = { key, base, vlm ->
                viewModel.updateSettings(key, base, vlm)
                showSettings = false
            }
        )
    }
}

@Composable
private fun FeedScreen(
    state: FeedUiState,
    onSourceChange: (FeedSource) -> Unit,
    onSubredditChange: (String) -> Unit,
    onRefresh: () -> Unit,
    onOpenPost: (PostItem) -> Unit
) {
    Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SourceSelector(source = state.source, onSourceChange = onSourceChange)
            OutlinedTextField(
                value = state.subreddit,
                onValueChange = onSubredditChange,
                modifier = Modifier.weight(1f),
                singleLine = true,
                label = { Text("Subreddit / Source key") }
            )
            Button(onClick = onRefresh) { Text("Load") }
        }
        Spacer(Modifier.height(8.dp))
        if (state.loading) {
            CircularProgressIndicator()
        } else if (state.error != null) {
            Text(state.error, color = MaterialTheme.colorScheme.error)
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(state.posts) { post ->
                    Card(onClick = { onOpenPost(post) }) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Text(post.data.title, style = MaterialTheme.typography.titleMedium)
                            Text(
                                "${post.data.author} • ${post.data.score} pts • ${post.data.numComments} comments",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SourceSelector(source: FeedSource, onSourceChange: (FeedSource) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    Column {
        OutlinedButton(onClick = { expanded = true }) {
            Text("Source: ${source.name}")
        }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            FeedSource.entries.forEach { option ->
                DropdownMenuItem(
                    text = { Text(option.name) },
                    onClick = {
                        onSourceChange(option)
                        expanded = false
                    }
                )
            }
        }
    }
}

@Composable
private fun DetailScreen(
    state: DetailUiState,
    onSummaryText: () -> Unit,
    onSummaryComments: () -> Unit,
    onSummaryArticle: () -> Unit,
    onPromptChange: (String) -> Unit,
    onAskAi: () -> Unit
) {
    val post = state.post ?: return
    Column(modifier = Modifier.fillMaxSize().padding(12.dp).verticalScroll(rememberScrollState())) {
        if (state.loading) {
            CircularProgressIndicator()
            return
        }
        if (state.error != null) {
            Text(state.error, color = MaterialTheme.colorScheme.error)
        }
        Text(post.data.title, style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(4.dp))
        Text("${post.data.author} • ${post.data.score} pts • ${post.data.numComments} comments")
        if (post.data.selftext.isNotBlank()) {
            Spacer(Modifier.height(8.dp))
            Text(post.data.selftext)
        }
        Spacer(Modifier.height(12.dp))
        Text("Comments", style = MaterialTheme.typography.titleMedium)
        FlattenedCommentList(comments = state.comments)
        Spacer(Modifier.height(12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            OutlinedButton(onClick = onSummaryText) { Text("Text") }
            OutlinedButton(onClick = onSummaryArticle) { Text("Article") }
            OutlinedButton(onClick = onSummaryComments) { Text("Comments") }
        }
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(
            value = state.aiPrompt,
            onValueChange = onPromptChange,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Ask AI about this post") }
        )
        Spacer(Modifier.height(8.dp))
        Button(onClick = onAskAi) { Text("Submit") }
        if (state.aiLoading) {
            Spacer(Modifier.height(8.dp))
            CircularProgressIndicator()
        }
        if (state.aiText.isNotBlank()) {
            Spacer(Modifier.height(8.dp))
            Text(state.aiText)
        }
    }
}

@Composable
private fun FlattenedCommentList(comments: List<CommentNode>) {
    val rows = remember(comments) { flattenCommentNodes(comments) }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        rows.forEach { row ->
            Text("${"  ".repeat(row.level)}${row.author}: ${row.body}")
        }
    }
}

private data class FlatComment(val level: Int, val author: String, val body: String)

private fun flattenCommentNodes(nodes: List<CommentNode>, level: Int = 0): List<FlatComment> {
    return buildList {
        nodes.forEach { node ->
            add(FlatComment(level, node.data.author, node.data.body))
            addAll(flattenCommentNodes(node.replies, level + 1))
        }
    }
}

@Composable
private fun SettingsDialog(
    state: SettingsUiState,
    onDismiss: () -> Unit,
    onSave: (apiKey: String, baseModel: String, vlmModel: String) -> Unit
) {
    var apiKey by remember(state.apiKey) { mutableStateOf(state.apiKey) }
    var baseModel by remember(state.baseModel) { mutableStateOf(state.baseModel) }
    var vlmModel by remember(state.vlmModel) { mutableStateOf(state.vlmModel) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Settings") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(
                    value = apiKey,
                    onValueChange = { apiKey = it },
                    label = { Text("OpenRouter API Key") },
                    modifier = Modifier.fillMaxWidth()
                )
                OutlinedTextField(
                    value = baseModel,
                    onValueChange = { baseModel = it },
                    label = { Text("Base LLM Model") },
                    modifier = Modifier.fillMaxWidth(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Text)
                )
                OutlinedTextField(
                    value = vlmModel,
                    onValueChange = { vlmModel = it },
                    label = { Text("VLM Model") },
                    modifier = Modifier.fillMaxWidth()
                )
            }
        },
        confirmButton = {
            Button(onClick = { onSave(apiKey, baseModel, vlmModel) }) { Text("Save") }
        },
        dismissButton = {
            OutlinedButton(onClick = onDismiss) { Text("Cancel") }
        }
    )
}
