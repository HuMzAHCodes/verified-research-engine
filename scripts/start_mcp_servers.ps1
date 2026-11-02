# Starts the project's local MCP servers, each in its own PowerShell window.
#
# CONCEPT: pdf-parser-mcp, open choice B (D-46). Both servers use HTTP on this
# machine and are started by hand; this script makes that one step. A server
# whose port is already in use is reported and left alone, so running the
# script twice does not start a second copy. Close a server's window, or press
# Ctrl+C in it, to stop it.
#
# Run from the project root:
#     .\scripts\start_mcp_servers.ps1

$projectRoot = Split-Path -Parent $PSScriptRoot

$mcpServers = @(
    @{ Name = "search-mcp"; Module = "mcp_servers.search_mcp.server"; Port = 8765 },
    @{ Name = "pdf-parser-mcp"; Module = "mcp_servers.pdf_parser_mcp.server"; Port = 8766 }
)

foreach ($mcpServer in $mcpServers) {
    $listening = Get-NetTCPConnection -LocalPort $mcpServer.Port -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        Write-Host "$($mcpServer.Name) is already running on port $($mcpServer.Port)"
        continue
    }
    # The window's title names the server, so the two windows are easy to tell apart.
    $serverCommand = "`$host.UI.RawUI.WindowTitle = '$($mcpServer.Name)'; " +
        "Set-Location '$projectRoot'; uv run python -m $($mcpServer.Module)"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $serverCommand
    Write-Host "Started $($mcpServer.Name) on port $($mcpServer.Port) in a new window"
}
