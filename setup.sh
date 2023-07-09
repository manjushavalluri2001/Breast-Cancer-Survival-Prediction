mkdir -p ~/.streamlit/

echo "[theme]
primaryColor = '#f9adcf'
font = 'sans serif'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
