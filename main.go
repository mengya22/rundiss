package main

import (
	"config"
	"fmt"
	"sync/atomic"
	"time"
	"webser"
)

type AppConfig struct {
	port int
	nginxAddr string
}

type AppconfigMgr struct {
	config atomic.Value
}

var appConfigMgr = &AppconfigMgr{}

func (a *AppconfigMgr) Callback (conf *config.Config) {
	var appConfig = &AppConfig{}
	port, err := conf.GetInt("server_port")
	if err != nil {
		fmt.Println("get port failed")
		return
	}
	appConfig.port = port
	fmt.Println("port:", appConfig.port)
	nginxAddr, err := conf.GetString("nginx_addr")
	if err != nil {return }
	appConfig.nginxAddr = nginxAddr
	fmt.Println("nginx addr : ", appConfig.nginxAddr)

	appConfigMgr.config.Store(appConfig)
}

func run() {
	for {
		appConfig := appConfigMgr.config.Load().(*AppConfig)
		fmt.Println("port:", appConfig.port)
		fmt.Println("nginx addr:", appConfig.nginxAddr)
		time.Sleep(5 * time.Second)
	}
}

/*func main() {
	conf, err := config.NewConfig("H:\\gopro\\mylunzi\\src\\config.conf")
	if err != nil {
		fmt.Println("parse config failed, err: ", err)
		return
	}
	conf.AddNotifyer(appConfigMgr)

	var appConfig = &AppConfig{}
	appConfig.port, err = conf.GetInt("server_port")
	fmt.Println("port---: ", appConfig.port)
	appConfig.nginxAddr, err = conf.GetString("nginx_addr")
	fmt.Println("nginx--- addr: ", appConfig.nginxAddr)
	appConfigMgr.config.Store(appConfig)
	run()


}*/


func main() {
	//webser.Start()
	
}
}
