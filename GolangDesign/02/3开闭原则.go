package main

import "fmt"

// 抽象的银行业务员
type AbstractBanker interface {
	DoBusi() // 抽象的业务处理接口
}

// 存款的业务员
type SaveBanker struct {
	// AbstractBanker
}

func (sb *SaveBanker) DoBusi() {
	fmt.Println("进行了存款")
}

// 转账的业务员
type TranferBanker struct {
	// AbstractBanker
}

func (tb *TranferBanker) Tranfer() {
	w
}

func main() {
	// 存款的业务
	sb := SaveBanker{}
	sb.DoBusi()
}
