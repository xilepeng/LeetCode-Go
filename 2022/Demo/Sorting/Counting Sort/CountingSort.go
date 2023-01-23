package main

import (
	"fmt"
	"math"
)

func counting_sort(A, Aux, sortedA []int, N int) {
	K := 0
	for i := 0; i < N; i++ {
		if K < A[i] {
			K = A[i]
		}
	}
	// for i := 0; i <= K; i++ {
	// 	Aux[i] = 0
	// }
	for i := 0; i < N; i++ {
		Aux[A[i]]++
	}
	j := 0
	for i := 0; i <= K; i++ {
		tmp := Aux[i]
		for ; tmp > 0; tmp-- {
			sortedA[j] = i
			j++
		}
	}
}

func main() {
	var n int

	fmt.Scanf("%d", &n)
	A, sortedA := make([]int, n), make([]int, n)
	max := math.MinInt64
	for i := 0; i < n; i++ {
		fmt.Scanf("%d", &A[i])
		if A[i] > max {
			max = A[i]
		}
	}
	Aux := make([]int, max+1)
	counting_sort(A, Aux, sortedA, n)
	for i := 0; i < n; i++ {
		tmp := Aux[sortedA[i]]
		if tmp > 0 {
			fmt.Printf("%d %d\n", sortedA[i], Aux[sortedA[i]])
		}
		for tmp -= 1; tmp > 0; i++ {
			tmp--
		}
	}
}
