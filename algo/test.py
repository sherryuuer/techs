def permutations(nums):
    def backtrack(start, end):
        if start == end:
            print(nums[:])  # 输出当前排列状态
            result.append(nums[:])
        else:
            for i in range(start, end):
                print(f"交换前：{nums}")
                # 交换元素
                nums[start], nums[i] = nums[i], nums[start]
                print(f"交换后：{nums}")
                # 递归调用
                backtrack(start + 1, end)
                print(f"递归返回后，恢复：{nums}")
                # 恢复原始状态，以便下一次迭代
                nums[start], nums[i] = nums[i], nums[start]

    result = []
    backtrack(0, len(nums))
    return result


# 示例
nums = [1, 2, 3]
result = permutations(nums)
print("全排列结果：", result)
