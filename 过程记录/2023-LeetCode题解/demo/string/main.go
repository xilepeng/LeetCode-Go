package main

import (
	"fmt"
	"strings"
)

func main() {
	s := "hello,world"
	new_s := strings.Split(s, ",")
	fmt.Println(new_s)
}
