package main

import "fmt"

// swap 不使用第3个变量交换
func swap(x, y *int) {
	*x ^= *y
	*y ^= *x
	*x ^= *y
	fmt.Println("交换后:", *x, *y)
}

func main() {
	x, y := 1, 2
	fmt.Println("交换前:", x, y)
	swap(&x, &y)
	fmt.Println("交换后:", x, y)
}

// 交换前: 1 2
// 交换后: 2 1
// 交换后: 2 1
