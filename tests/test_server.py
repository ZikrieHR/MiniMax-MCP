import importlib
import os
import sys

import anyio


os.environ.setdefault("MINIMAX_API_KEY", "test-api-key")
os.environ.setdefault("MINIMAX_API_HOST", "https://api.example.invalid")

Client = importlib.import_module("mcp").Client
StdioServerParameters = importlib.import_module("mcp").StdioServerParameters
stdio_client = importlib.import_module("mcp.client.stdio").stdio_client
server = importlib.import_module("minimax_mcp.server")


EXPECTED_TOOLS = {
    "text_to_audio",
    "list_voices",
    "voice_clone",
    "play_audio",
    "generate_video",
    "query_video_generation",
    "text_to_image",
    "voice_design",
}


def test_initialize_and_list_tools():
    async def exercise_server():
        async with Client(server.mcp, raise_exceptions=True) as client:
            assert client.server_info.name == "Minimax"
            assert client.protocol_version

            result = await client.list_tools()
            assert {tool.name for tool in result.tools} == EXPECTED_TOOLS

    anyio.run(exercise_server)


def test_stdio_startup_and_list_tools():
    async def exercise_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "minimax_mcp.server"],
            env={
                "MINIMAX_API_KEY": "test-api-key",
                "MINIMAX_API_HOST": "https://api.example.invalid",
            },
        )
        for mode in ("auto", "legacy"):
            async with Client(stdio_client(parameters), mode=mode) as client:
                result = await client.list_tools()
                assert {tool.name for tool in result.tools} == EXPECTED_TOOLS
                if mode == "legacy":
                    assert client.protocol_version != "2026-07-28"

    anyio.run(exercise_stdio_server)


def test_tool_call_uses_mocked_api(monkeypatch):
    def fake_post(path, **kwargs):
        assert path == "/v1/get_voice"
        return {
            "system_voice": [{"voice_name": "Test Voice", "voice_id": "test-voice"}],
            "voice_cloning": [],
        }

    monkeypatch.setattr(server.api_client, "post", fake_post)

    async def exercise_tool():
        async with Client(server.mcp, raise_exceptions=True) as client:
            result = await client.call_tool("list_voices", {"voice_type": "system"})
            assert result.is_error is False
            assert "test-voice" in result.content[0].text

    anyio.run(exercise_tool)


def test_main_runs_without_stdout_protocol_noise(monkeypatch, capsys):
    run_called = False

    def fake_run():
        nonlocal run_called
        run_called = True

    monkeypatch.setattr(server.mcp, "run", fake_run)
    server.main()

    assert run_called is True
    assert capsys.readouterr().out == ""
