package main

import "fmt"

type Banker struct{}

// 存款业务
func (b *Banker) Save() {
	fmt.Println("进行了存款业务...")
}

// 转账业务
func (b *Banker) Tansfer() {
	fmt.Println("进行了转账业务...")
}

// 支付业务
func (b *Banker) Pay() {
	fmt.Println("进行了支付业务...")
}

// 添加股票业务 +
func (b *Banker) Shares() {
	fmt.Println("进行了股票业务...")
}

func main() {
	banker := &Banker{}

	banker.Save()
	banker.Tansfer()
	banker.Pay()
}
