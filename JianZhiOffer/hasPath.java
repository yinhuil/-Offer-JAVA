package JianZhiOffer;

public class hasPath {
/*	题目描述
	请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
	路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，
	向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该
	格子。 例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路
	径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个
	格子之后，路径不能再次进入该格子。*/

//	利用回溯的方法，先找到一个的时候，然后进入searchFromHere函数，并且用visited这个函数来表示是否已经用过。
	public static boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
		boolean[] visited = new boolean[matrix.length];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				if (searchFromHere(matrix, rows, cols, i, j, 0, str, visited))
					return true;
			}
		}
		return false;
	}

	public static boolean searchFromHere(char[] matrix, int rows, int cols,
			int r, int c, int index, char[] str, boolean[] visited) {
		if (r < 0 || r >= rows || c < 0 || c >= cols
				|| matrix[r * cols + c] != str[index] || visited[r * cols + c])
			return false;
		if (index == str.length - 1)
			return true;
		visited[r * cols + c] = true;
		if (searchFromHere(matrix, rows, cols, r - 1, c, index + 1, str,
				visited)
				|| searchFromHere(matrix, rows, cols, r, c - 1, index + 1, str,
						visited)
				|| searchFromHere(matrix, rows, cols, r + 1, c, index + 1, str,
						visited)
				|| searchFromHere(matrix, rows, cols, r, c + 1, index + 1, str,
						visited))
			return true;
		visited[r * cols + c] = false;
		return false;
	}

	public static void main(String[] args) {
		char[] str1 = { 'a', 'b', 'c', 'e', 's', 'f', 'c', 's', 'a', 'd', 'e',
				'e' };
		char[] str2 = { 'b', 'c', 'c', 'e', 'd' };
		System.out.println(hasPath(str1, 3, 4, str2));

	}

}
