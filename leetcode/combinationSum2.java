package leetcodeForJianzhi;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
/*Given a collection of candidate numbers (C) and a target number (T), 
find all unique combinations in C where the candidate numbers sums to T.

Each number in C may only be used once in the combination.

Note:
All numbers (including target) will be positive integers.
The solution set must not contain duplicate combinations.
For example, given candidate set [10, 1, 2, 7, 6, 1, 5] and target 8, 
A solution set is: 
[
  [1, 7],
  [1, 2, 5],
  [2, 6],
  [1, 1, 6]
]*/


//解析，因为首先保证不能让一个值出现两次，所以相对于combationsum1来说，这里得dfs(nums, target-nums[i], result, list, i+1);要为i+1
//   第二，因为不能出现整体的重复数组， 所以需要提前将数组进行排序，然后如果发现相同的跳过

public class combinationSum2 {
	public static List<List<Integer>> combinationSum2(int[] nums, int target) {
		List<List<Integer>> result=new ArrayList<List<Integer>>();
		Arrays.sort(nums);
		List<Integer> list=new ArrayList<Integer>();
		dfs(nums,target,result,list,0);
		return result;
	}

	private static void dfs(int[] nums, int target, List<List<Integer>> result,
			List<Integer> list, int start) {
		if(target<0) return ;
		else if(target==0) result.add(new ArrayList<Integer>(list));
		else{
			for(int i=start;i<nums.length;i++){
				if (i > start && nums[i] == nums[i - 1])
					continue; // skip duplicates
				list.add(nums[i]);
				dfs(nums, target-nums[i], result, list, i+1);
				list.remove(list.size()-1);
			}
		}
		
	}
	public static void main(String[] args) {
		int[] in={10,1,2,7,6,1,5};
		List<List<Integer>> result=combinationSum2(in, 8);
		for (List<Integer> list : result) {
			System.out.println(list);
		}
	}
}
















