package main

import "fmt"

func main() {
	Arr := []int{1}
	Arr[0] += '0'
	fmt.Println("Arr=", Arr)

	A := make([]byte, 5)
	A[0] += '0'
	fmt.Println("A=", A)
}
