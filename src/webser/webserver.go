package webser

import (
	"handler"
	"log"
	"net/http"
)


type MyMux struct{
}

func (p *MyMux)ServeHTTP(w http.ResponseWriter, r *http.Request){
	if r.URL.Path == "/"{
		handler.IndexHandler(w, r)
		return
	}
	if r.URL.Path == "/about"{
		handler.About(w, r)
		return
	}
	if r.URL.Path == "/upload"{
		handler.UploadHandler(w,r)
		return
	}
	if r.URL.Path == "/login"{
		handler.LoginHandler(w,r)
		return
	}
	if r.URL.Path == "/myget"{
		handler.MyGetHandler(w,r)
		return
	}
	http.NotFound(w,r)
	return
}

func Start() {
	myMux := &MyMux{}
	//http.HandleFunc("/", webHandler)
	//http.HandleFunc("/login", loginHandler)
	err := http.ListenAndServe(":9090", myMux)
	if err != nil {
		log.Fatal("ListenAndServer: ", err)
	}
}
