package leetcodeForJianzhi;

public class candy {
/*	题目描述

	There are N children standing in a line. Each child is assigned a rating value.
	You are giving candies to these children subjected to the following requirements:
	Each child must have at least one candy.
	Children with a higher rating get more candies than their neighbors.
	What is the minimum candies you must give?*/

//	题解，先从头开始遍历，当发现后面比前面大的话就给他加一，然后在从后往前遍历，当ratings[i] > ratings[i + 1]的时候
//	这个时候需要注意了，不仅需要考虑后面的还要考虑前面的，因此就是Math.max(dp[i], dp[i + 1] + 1)
	public int candy(int[] ratings) {
		int lens = ratings.length;
		int[] dp = new int[lens];
		dp[0] = 1;
		for (int i = 1; i < lens; i++) {
			if (ratings[i] > ratings[i - 1]) {
				dp[i] = dp[i - 1] + 1;
			} else {
				dp[i] = 1;
			}
		}
		for (int i = lens - 2; i >= 0; i--) {
			if (ratings[i] > ratings[i + 1]) {
				dp[i] = Math.max(dp[i], dp[i + 1] + 1);
			}
		}
		int sum = 0;
		for (int i = 0; i < lens; i++) {
			sum += dp[i];
		}
		return sum;
	}
}
