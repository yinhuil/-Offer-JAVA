package leetcodeForJianzhi;

public class bestTimetoSellIV {
	
/*	Say you have an array for which the ith element is the price of a given stock on day i.

	Design an algorithm to find the maximum profit. You may complete at most k transactions.

	Note:
	You may not engage in multiple transactions at the same time (ie, you must sell the stock 
			before you buy again).*/
	
//解析：这里面当k的值大于n的一半时，也就是随意买卖，无限多次的买卖
//	当k小于n的一半的情况
//	这时候定一个二维的dp，其中localMoney表示现在所有的钱，dp表示挣的钱
//	第一行的时候，首先没有钱，买了一个prices[0]
//	着重理解localMoney=Math.max(localMoney, dp[i-1][j-1]-prices[j]);
//	其中dp[i-1][j-1]-prices[j]表示上一轮交易中，自己腰包的钱，和要买当前东西的钱。
	public static int maxProfit(int k, int[] prices) {
		int n=prices.length;
		if(n<=1) return 0;
		if(k>=n/2){
			int max_profit=0;
			for(int i=1;i<prices.length;i++){
				if(prices[i]-prices[i-1]>0){
					max_profit+=prices[i]-prices[i-1];
				}
			}
			return max_profit;
		}
		int[][] dp=new int[k+1][prices.length];
		for(int i=1;i<=k;i++){
			int localMoney=dp[i-1][0]-prices[0];
			for(int j=1;j<prices.length;j++){
				dp[i][j]=Math.max(dp[i][j-1], prices[j]+localMoney);
				localMoney=Math.max(localMoney, dp[i-1][j-1]-prices[j]);
			}
		}
		return dp[k][n-1];
	}
	public static void main(String[] args) {
		int[] in={1,6,5,3,2,7};
		System.out.println(maxProfit(2, in));
	}
}
