go install go.k6.io/xk6/cmd/xk6@latest
$(go env GOPATH)/bin/xk6 build --with github.com/phymbert/xk6-sse@latest

# run with:
# $ ./k6 run --vus 1 --duration 600s script.js