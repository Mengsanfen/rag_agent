from mcp_server.tools_server import server




if __name__ == '__main__':
    server.run(
        transport='streamable-http',
        host='0.0.0.0', 
        port=5000, 
        log_level='debug', 
        path='/streamable'
    )