package config

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

var a int
type Config struct {
	 filename string
	 lastModifyTime int64
	 data map[string]string
	 rwLock sync.RWMutex
	 notifyList []Notifyer
}
func NewConfig(filename string) (conf *Config, err error) {
	conf =&Config{
		filename:filename,
		data:make(map[string]string, 1024),
	}
	m, err := conf.parse()
	if err != nil {return }
	conf.rwLock.Lock()
	conf.data = m
	conf.rwLock.Unlock()
	a = 3
	go conf.reload()
	return
}

func (c *Config) AddNotifyer(n Notifyer) {
	c.notifyList = append(c.notifyList, n)
}
/**
  *read config.txt
  *@input
  *@output: map
  */
func (c *Config) parse() (m map[string]string,err error) {
	m = make(map[string]string, 1024)
	file, err := os.Open(c.filename)
	if err != nil {return }
	var lineNo int
	reader := bufio.NewReader(file)
	for {
		line, errRet := reader.ReadString('\n')
		if errRet == io.EOF {break}
		if errRet != nil {
			err = errRet
			return
		}
		lineNo++
		line = strings.TrimSpace(line)
		if len(line) == 0 || line[0] == '\n' || line[0] == '+' || line[0] == ';' {
			continue
		}
		arr := strings.Split(line,"=")
		if len(arr) == 0 {
			fmt.Printf("invalid config, line:d\n", lineNo)
			continue
		}
		key := strings.TrimSpace(arr[0])
		value := strings.TrimSpace(arr[1])
		m[key] = value
	}
	return
}

func (c *Config) reload() {
	ticker := time.NewTicker(time.Second * 5)
	for _ = range ticker.C {
		func() {
			fmt.Println("a:",a)
			file, err :=os.Open(c.filename)
			if err != nil {
				fmt.Printf("open %s failed,err: %v \n",c.filename, err)
				return
			}
			defer file.Close()
			fileInfo, err := file.Stat()
			if err != nil {
				fmt.Printf("stat %s failed, err:%v\n",c.filename, err)
				return
			}
			curModifyTime := fileInfo.ModTime().Unix()
			fmt.Printf("%v-----%v\n",curModifyTime, c.lastModifyTime)
			if curModifyTime > c.lastModifyTime{
				m, err := c.parse()
				if err != nil {
					fmt.Print("parse faile,err:")
					return
				}
				c.rwLock.Lock()
				c.data = m
				c.rwLock.Unlock()
				for _, n := range c.notifyList {
					n.Callback(c)
				}
				c.lastModifyTime = curModifyTime
			}
		}()
	}
}

func (c *Config) GetInt(key string)(value int,err error){
	// 根据int获取
	c.rwLock.RLock()
	defer c.rwLock.RUnlock()
	str,ok:=c.data[key]
	if !ok{
		err = fmt.Errorf("key[%s] not found",key)
		return
	}
	value,err = strconv.Atoi(str)
	return
}

func (c *Config) GetString(key string)(value string,err error){
	// 根据字符串获取
	c.rwLock.RLock()
	defer c.rwLock.RUnlock()
	value,ok := c.data[key]
	if !ok{
		err = fmt.Errorf("key[%s] not found",key)
		return
	}
	return
}