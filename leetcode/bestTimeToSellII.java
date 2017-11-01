package leetcodeForJianzhi;

public class bestTimeToSellII {
/*	Say you have an array for which the ith element is the price of a given stock on day i.

	Design an algorithm to find the maximum profit. You may complete as many transactions 
	as you like (ie, buy one and sell one share of the stock multiple times). However,
	you may not engage in multiple transactions at the same time (ie, you must sell the
			stock before you buy again).*/
	
	public int maxProfit(int[] prices) {
		if(prices.length==0) return 0;
		int max_profit=0;
		int pre_profit=0;
		int min_price=prices[0];
		for(int i=1;i<prices.length;i++){
			int cur_profit=prices[i]-min_price;
			if(cur_profit<pre_profit){
				max_profit+=pre_profit;
				pre_profit=0;
				min_price=prices[i];
			}else{
				pre_profit=cur_profit;
			}
		}
		max_profit+=pre_profit;
		return max_profit;
	}
//	第二种方法，只要发现就买卖
	public int maxpro(int[] prices){
		int max_profit=0;
		for(int i=1;i<prices.length;i++){
			if(prices[i]-prices[i-1]>0){
				max_profit+=prices[i]-prices[i-1];
			}
		}
		return max_profit;
	}
}
