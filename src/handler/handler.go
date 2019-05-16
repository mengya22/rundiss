package handler

import (
	"bufio"
	"encoding/json"
	"execmachine"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"strings"
	"html/template"
)


func UploadHandler(writer http.ResponseWriter, request *http.Request) {
	request.ParseMultipartForm(32<<20)
	//接收客户端传来的文件 uploadfile 与客户端保持一致
	file, handler, err := request.FormFile("uploadfile")
	fmt.Println("file :", file)
	if err != nil{
		fmt.Println(err)
		return
	}
	defer file.Close()
	//write file
	buf := bufio.NewReader(file)
	for {
		line, err := buf.ReadString('\n')
		line = strings.TrimSpace(line)
		fmt.Println(line)
		if err != nil {
			if err == io.EOF {
				break
			}
		}
	}


	//上传的文件保存在ppp路径下
	ext := path.Ext(handler.Filename)       //获取文件后缀
	fileNewName := "lay"+ext

	f, err := os.OpenFile("H:\\gopro\\mylunzi\\src\\ppp\\"+fileNewName, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil{
		fmt.Println(err)
		return
	}
	defer f.Close()

	io.Copy(f, file)

	fmt.Fprintln(writer, "upload ok!"+fileNewName)
}

func IndexHandler(writer http.ResponseWriter, request *http.Request) {
	const tpl = `<html>
				<head>
				<title>上传文件</title>
				</head>
				<body>
				<form enctype="multipart/form-data" action="/upload" method="post">
				<input type="file" name="uploadfile">
				<input type="hidden" name="token" value="{...{.}...}">
				<input type="submit" value="upload">
				</form>
				</body>
				</html>`
	writer.Write([]byte(tpl))
}

func LoginHandler(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	fmt.Println("method: ",r.Method)
	if r.Method == "GET" {
		t, _ := template.ParseFiles(".\\view\\login.ctpl")
		t.Execute(w, nil)
	}else if r.Method == "POST" {
		fmt.Println("username: ",r.Form["username"])
		fmt.Println("password: ", r.Form["password"])
	}
}

func WebHandler(w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	fmt.Println("start------")
	fmt.Println(r.Form)
	fmt.Println("path: ", r.URL.Path)
	fmt.Println("scheme: ", r.URL.Scheme)
	fmt.Println(r.Form["url_long"])
	for k, v := range r.Form{
		fmt.Println("key: ",k)
		fmt.Println("val: ",strings.Join(v,"\n"))
	}
	fmt.Println("end-----")
	fmt.Fprintf(w,"hello chain")
}

func About(w http.ResponseWriter, r *http.Request){
	fmt.Fprintf(w, "i am chain, from shanghai")
}

/**
 *get request 
 *if sn == run, run a python code 	
 */
func MyGetHandler(w http.ResponseWriter, r *http.Request) {
	// parse query parameter
	vals := r.URL.Query()
	param, _ := vals["sn"]  // get query parameters
	fmt.Println(param)
	if param[0] == "run" {
	resu := execmachine.Execpython()
	// composite response body
	var res = map[string]string{"result":resu, "name":param[0]}
	response, _ := json.Marshal(res)
	w.Header().Set("Content-Type", "application/json")
	w.Write(response)
	} else {
		var res = map[string]string{"result":"no run", "name":param[0]}
		response, _ := json.Marshal(res)
		w.Header().Set("Content-Type", "application/json")
		w.Write(response)
	}

}
