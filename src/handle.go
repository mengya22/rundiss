package main

import (
	"encoding/json"
	"net/http"
)

func MyGetHandler(w http.ResponseWriter, r *http.Request) {
	// parse query parameter
	vals := r.URL.Query()
	param, _ := vals["servicename"]  // get query parameters
	// composite response body
	var res = map[string]string{"result":"succ", "name":param[0]}
	response, _ := json.Marshal(res)
	w.Header().Set("Content-Type", "application/json")
	w.Write(response)
}