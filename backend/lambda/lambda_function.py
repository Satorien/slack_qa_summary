import os
import json
import boto3
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# ------------------------------------------------------------------------------------
# 初期化処理
# Lambdaハンドラの外で初期化することで、関数のウォームスタート時に再利用され効率的です。
# ------------------------------------------------------------------------------------

try:
    slack_bot_token = os.environ["SLACK_BOT_TOKEN"]
except KeyError:
    raise Exception("環境変数 `SLACK_BOT_TOKEN` が設定されていません。")

# SlackとBedrockのクライアントを初期化
slack_client = WebClient(token=slack_bot_token)
# Bedrockが利用可能なリージョンを指定（例: us-east-1）
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')


def get_latest_message_from_slack(channel_id: str) -> str | None:
    """指定されたSlackチャンネルから最新のメッセージを1件取得する"""
    print(f"Fetching message from channel: {channel_id}")
    try:
        response = slack_client.conversations_history(channel=channel_id, limit=1, inclusive=True)
        if response["messages"]:
            latest_message = response["messages"][0]["text"]
            print(f"Found message: {latest_message}")
            return latest_message
        else:
            print("No messages found in the channel.")
            return None
    except SlackApiError as e:
        print(f"Slack API Error (history): {e.response['error']}")
        raise


def invoke_bedrock_model(text_to_summarize: str) -> str:
    """BedrockのClaude 3.7 Sonnetモデルを呼び出し、テキストを要約する"""
    print("Invoking Bedrock model (Claude 3.7 Sonnet)...")
    prompt = f"""Human: 以下のSlackの投稿を、重要なポイントを箇条書きで簡潔に要約してください。

<投稿内容>
{text_to_summarize}
</投稿内容>

Assistant:"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    })

    try:
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId='anthropic.claude-3-7-sonnet-20250219-v1:0',  # Claude 3.7 Sonnet
            accept='application/json',
            contentType='application/json',
        )
        response_body = json.loads(response.get('body').read())
        summary = response_body.get('content')[0].get('text')
        print(f"Successfully got summary from Bedrock: {summary}")
        return summary
    except Exception as e:
        print(f"Bedrock invocation error: {e}")
        raise


def post_message_to_slack(channel_id: str, text: str):
    """指定されたSlackチャンネルにメッセージを投稿する"""
    print(f"Posting message to channel: {channel_id}")
    try:
        slack_client.chat_postMessage(channel=channel_id, text=text)
        print("Successfully posted message to Slack.")
    except SlackApiError as e:
        print(f"Slack API Error (post): {e.response['error']}")
        raise

# ------------------------------------------------------------------------------------
# Lambdaハンドラ関数
# この関数が、Lambda実行時に最初に呼び出されます。
# ------------------------------------------------------------------------------------
def lambda_handler(event, context):
    """メインの処理関数"""
    
    # PoCのため、チャンネルIDをハードコード
    # TODO: 自分のSlackチャンネルIDに置き換えてください
    INPUT_CHANNEL_ID = "C091S67LDEJ"
    OUTPUT_CHANNEL_ID = "C09344CE82C"
    
    print("--- Start Execution ---")
    
    try:
        # 1. Slackから最新メッセージを取得
        message_text = get_latest_message_from_slack(INPUT_CHANNEL_ID)
        
        if not message_text:
            print("--- End Execution: No message found ---")
            return {'statusCode': 200, 'body': json.dumps('No new messages to process.')}

        # 2. Bedrockで要約
        summary = invoke_bedrock_model(message_text)

        # 3. 結果をSlackに投稿
        result_text = f"【AIによる要約】\n>>> {summary}"
        post_message_to_slack(OUTPUT_CHANNEL_ID, result_text)
        
        print("--- End Execution: Success ---")
        return {'statusCode': 200, 'body': json.dumps('Successfully processed message.')}
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("--- End Execution: Error ---")
        # エラーが発生した場合、500エラーを返す
        return {'statusCode': 500, 'body': json.dumps(f'An error occurred: {str(e)}')}