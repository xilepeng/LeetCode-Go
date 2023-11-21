package main

import (
	"fmt"
	"strconv"
)

func main() {
	nums := []int{1, 2, 3}
	s := []string{}
	num := []int{}
	for _, n := range nums {
		s = append(s, strconv.Itoa(n))
	}
	fmt.Println("整数转字符串", s) //字符串 [1 2 3] 数据类型：string

	for i := 0; i < len(s); i++ {
		n, _ := strconv.Atoi(s[i])
		num = append(num, n)
	}
	fmt.Println("字符串转整数", num) //整数 [1 2 3]
}
