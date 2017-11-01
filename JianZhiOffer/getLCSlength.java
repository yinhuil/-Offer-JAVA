package JianZhiOffer;

public class getLCSlength {
//	利用动态规划来进行预算，有三种情况，1，当两个子字符相同是 dp[i][j]=dp[i-1][j-1]+1 2,
//	当不相等的时候 dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1])
//	但是首先需要处理第一行和第一列的情况，在通常情况下就让dp多增加一行和一列，然后将第一行和第一列变成0；
	public static int[][] LCS(String str1, String str2) {
		int[][] opt = new int[str2.length() + 1][str1.length() + 1];

		for (int i = 0; i <= str2.length(); i++) {
			opt[i][0] = 0;
		}

		for (int j = 0; j <= str1.length(); j++) {
			opt[0][j] = 0;
		}

		for (int j = 1; j <= str1.length(); j++) {
			for (int i = 1; i <= str2.length(); i++) {
				if (str2.charAt(i - 1) == str1.charAt(j - 1)) {
					opt[i][j] = opt[i - 1][j - 1] + 1;
				} else {
					opt[i][j] = (opt[i - 1][j] >= opt[i][j - 1] ? opt[i - 1][j]
							: opt[i][j - 1]);
				}
			}
		}

		return opt;
	}

	public static void print(int[][] opt, String X, String Y, int i, int j) {

		if (i == 0 || j == 0) {
			return;
		}

		if (X.charAt(i - 1) == Y.charAt(j - 1)) {
			
			print(opt, X, Y, i - 1, j - 1);
			System.out.print(X.charAt(i - 1));

		} else if (opt[i - 1][j] >= opt[i][j]) {
			print(opt, X, Y, i - 1, j);
		} else {
			print(opt, X, Y, i, j - 1);
		}
	}
	
	public static void main(String[] args) {
		String str1="abcbdab";
		String str2="bcda";
		int[][] opt=LCS(str1, str2);
		print(opt, str2, str1, str2.length(), str1.length());
	}
}
