package leetcodeForJianzhi;

public class uniquePathsWithObstacles {
	
/*	题目描述

	Follow up for "Unique Paths":
	Now consider if some obstacles are added to the grids. How many unique paths would there be?
	An obstacle and empty space is marked as1and0respectively in the grid.
	For example,
	There is one obstacle in the middle of a 3x3 grid as illustrated below.
	[
	  [0,0,0],
	  [0,1,0],
	  [0,0,0]
	]
	The total number of unique paths is2.*/
	
//首先初始化第一行和第一列，如果发现当前位置为1，那么后面其余的不用赋值了，直接默认的0，因为如果发现当前道路堵住， 那么后面都是不通的
//	然后开始往右和往下遍历，如果发现位置为1，则直接赋值为0，其余，还是上面和左边的和
	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		int row=obstacleGrid.length;
		int cols=obstacleGrid[0].length;
		int[][] dp=new int[row][cols];
		for(int i=0;i<row;i++){
			if(obstacleGrid[i][0]==1){
				break;
			}
			dp[i][0]=1;
		}
		for(int j=0;j<cols;j++){
			if(obstacleGrid[0][j]==1) break;
			dp[0][j]=1;
		}
		for(int i=1;i<row;i++){
			for(int j=1;j<cols;j++){
				if(obstacleGrid[i][j]==1) dp[i][j]=0;
				else{
					dp[i][j]=dp[i-1][j]+dp[i][j-1];
				}
			}
		}
		return dp[row-1][cols-1];
	}
}
