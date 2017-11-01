package leetcodeForJianzhi;

import java.util.ArrayList;

public class minimumTotal {
/*	题目描述

	Given a triangle, find the minimum path sum from top to bottom. Each 
	step you may move to adjacent numbers on the row below.
	For example, given the following triangle
	[
	     [2],
	    [3,4],
	   [6,5,7],
	  [4,1,8,3]
	]

	The minimum path sum from top to bottom is11(i.e., 2 + 3 + 5 + 1 = 11).
	Note: 
	Bonus point if you are able to do this using only O(n) extra space, where
	n is the total number of rows in the triangle.*/
	
// 这道题的方法特别奇妙，从最下面一层开始，分别找到相邻的最小值加到上面的一层的值上面，依次循环，一直到第一层。   	
	public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
		for (int i = triangle.size() - 2; i >= 0; i--) {
			for (int j = 0; j < triangle.get(i + 1).size() - 1; j++) {
				int min = Math.min(triangle.get(i + 1).get(j),
						triangle.get(i + 1).get(j + 1));
				triangle.get(i).set(j, triangle.get(i).get(j) + min);
			}
		}
		return triangle.get(0).get(0);
	}
}
