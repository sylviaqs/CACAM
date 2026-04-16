set_proxy() {
  local PORT=${1:-7897}
  local PROXY="http://127.0.0.1:$PORT"
  git config --global http.proxy $PROXY
  git config --global https.proxy $PROXY
  export http_proxy=$PROXY
  export https_proxy=$PROXY
  for VAR in http_proxy https_proxy; do
    if grep -q "^export ${VAR}=" ~/.zshrc; then
      sed -i '' "s|^export ${VAR}=.*|export ${VAR}=${PROXY}|" ~/.zshrc
    else
      echo "export ${VAR}=${PROXY}" >> ~/.zshrc
    fi
  done

  echo "✅ 完成！"
}

set_proxy 6789
source ~/.zshrc