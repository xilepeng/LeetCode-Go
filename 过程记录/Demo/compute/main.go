package main

import "fmt"

func main() {
	fmt.Println("1+2<<1 =", 1+2<<1)
	fmt.Println("2*2<<1 =", 2*2<<1)
	fmt.Println("2<<1+1 =", 2<<1+1)
}

// 单目运算符
// 1+2<<1 = 5
// 2<<1+1 = 5

// << 左移乘以2的几次方    2<<2 2*2^2
// >> 右移除以2的几次方    2>>2 2/2^2
