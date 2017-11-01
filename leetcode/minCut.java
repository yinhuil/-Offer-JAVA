package leetcodeForJianzhi;

public class minCut {
/*	题目描述

	Given a string s, partition s such that every substring of the partition is a palindrome.
	Return the minimum cuts needed for a palindrome partitioning of s.
	For example, given s ="aab",
	Return1since the palindrome partitioning["aa","b"]could be produced using 1 cut.*/
	
/*	解析：这道题用动态规划， 首先定义一个跟s一样长度的数组，数组的每一个值表示这个位置的最小切割次数。   进入循环以后
	首先判断这个值从头到尾是不是一个回文序列，如果是，那就继续，如果不是，那就先设定一个最大的值i（也就是全部不一样的情况）
	，然后来向前遍历，如果发现i到前面的j是回文序列（因为从0到i从开始已经判断过了，所以j从1开始），
	就来判断Math.min(dp[i], dp[j - 1] + 1)，如果不是的话，
	就相当于这中间全部是不一样的字符，也就是Math.min(dp[i], dp[j - 1] + i + 1 - j)；*/
	public static int minCut(String s) {
		int[] dp = new int[s.length()];
		for (int i = 0; i < s.length(); i++) {
			dp[i] = isPalindrome(s.substring(0, i + 1)) ? 0 : i;
			if (dp[i] == 0)
				continue;
			for (int j = 1; j <= i; j++) {
				if (isPalindrome(s.substring(j, i + 1)))
					dp[i] = Math.min(dp[i], dp[j - 1] + 1);
				else
					dp[i] = Math.min(dp[i], dp[j - 1] + i + 1 - j);
			}
		}
		return dp[dp.length - 1];
	}

	// 判断回文
	public static boolean isPalindrome(String s) {
		boolean flag = true;
		for (int i = 0, j = s.length() - 1; i < j; i++, j--) {
			if (s.charAt(i) != s.charAt(j)) {
				flag = false;
				break;
			}
		}
		return flag;
	}
}
