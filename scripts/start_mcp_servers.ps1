# Starts the project's local MCP servers, each in its own PowerShell window,
# and returns only when they are ready to answer.
#
# CONCEPT: pdf-parser-mcp, open choice B (D-46). Both servers use HTTP on this
# machine and are started by hand; this script makes that one step. A server
# whose port is already in use is reported and left alone, so running the
# script twice does not start a second copy. Close a server's window, or press
# Ctrl+C in it, to stop it.
#
# CONCEPT: document-ingestion, "What broke". Start-Process returns as soon as
# the window opens, but a server needs a few seconds (uv, imports) before it
# listens. A command run straight after the script found nothing on the port.
# So the script now waits until each port is listening, up to a limit, and
# says which server is not ready if one never starts.
#
# Run from the project root:
#     .\scripts\start_mcp_servers.ps1

$projectRoot = Split-Path -Parent $PSScriptRoot
$readyTimeoutSeconds = 60

$mcpServers = @(
    @{ Name = "search-mcp"; Module = "mcp_servers.search_mcp.server"; Port = 8765 },
    @{ Name = "pdf-parser-mcp"; Module = "mcp_servers.pdf_parser_mcp.server"; Port = 8766 }
)

function Test-PortListening([int]$port) {
    $listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    return [bool]$listening
}

foreach ($mcpServer in $mcpServers) {
    if (Test-PortListening $mcpServer.Port) {
        Write-Host "$($mcpServer.Name) is already running on port $($mcpServer.Port)"
        continue
    }
    # The window's title names the server, so the two windows are easy to tell apart.
    $serverCommand = "`$host.UI.RawUI.WindowTitle = '$($mcpServer.Name)'; " +
        "Set-Location '$projectRoot'; uv run python -m $($mcpServer.Module)"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $serverCommand
    Write-Host "Started $($mcpServer.Name) on port $($mcpServer.Port) in a new window"
}

# Wait for every server, started now or earlier, to listen on its port.
$allReady = $true
foreach ($mcpServer in $mcpServers) {
    $waitedSeconds = 0
    while (-not (Test-PortListening $mcpServer.Port) -and $waitedSeconds -lt $readyTimeoutSeconds) {
        Start-Sleep -Seconds 1
        $waitedSeconds += 1
    }
    if (Test-PortListening $mcpServer.Port) {
        Write-Host "$($mcpServer.Name) is ready on port $($mcpServer.Port)"
    } else {
        Write-Host "$($mcpServer.Name) is not listening after $readyTimeoutSeconds s; read the error in its window"
        $allReady = $false
    }
}
if (-not $allReady) {
    exit 1
}
