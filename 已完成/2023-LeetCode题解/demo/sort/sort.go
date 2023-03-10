package main

import (
	"fmt"
	"sort"
)

func main() {
	A := []int{3, 2, 1}
	sort.Ints(A)
	fmt.Println(A)
}
