package leetcodeForJianzhi;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
/*Given a set of candidate numbers (C) (without duplicates) and a target number
(T), find all unique combinations in C where the candidate numbers sums to T.

The same repeated number may be chosen from C unlimited number of times.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
For example, given candidate set [2, 3, 6, 7] and target 7, 
A solution set is: 
[
  [7],
  [2, 2, 3]
]*/

//题解： 本题利用深度优先的方法，
public class CombinationSum {
	public static List<List<Integer>> combinationSum(int[] nums, int target) {
		List<List<Integer>> result = new ArrayList<List<Integer>>();
		Arrays.sort(nums);
		List<Integer> list = new ArrayList<Integer>();
		dfs(nums, target, result, list, 0);
		return result;
	}

	private static void dfs(int[] nums, int target, List<List<Integer>> result,
			List<Integer> list, int start) {
		if(target<0) return;
		else if(target==0) result.add(new ArrayList<Integer>(list));
		else{
			for(int i=start;i<nums.length;i++){
				list.add(nums[i]);
				dfs(nums, target-nums[i], result, list, i);
				list.remove(list.size()-1);
			}
		}
		
	}
	public static void main(String[] args) {
		int[] in={2,3,6,7};
		List<List<Integer>>result=combinationSum(in, 7);
		for (List<Integer> list : result) {
			System.out.println(list);
		}
	}
}
