import os

# 指定总目录
root_directory = '/XXXX/active-learning-master/experiments/020_revision'

# 定义一个函数来处理.sh文件
def process_sh_file(sh_file_path):
    with open(sh_file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    skip_next_line = False

    for line in lines:
        # 如果当前行以#SBATCH开头，跳过当前段落
        if line.strip().startswith('#SBATCH'):
            skip_next_line = True
        # 如果当前行不以#SBATCH开头但需要删除上一行
        elif skip_next_line:
            skip_next_line = False
            continue
        # 否则将当前行添加到新的内容中
        else:
            new_lines.append(line)

    # 如果最后一行是'wait'，删除该行
    if new_lines and new_lines[-1].strip() == 'wait':
        new_lines.pop()

    # 将新内容写回文件中
    with open(sh_file_path, 'w') as file:
        file.writelines(new_lines)

# 遍历总目录及其子目录中的.sh文件
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.sh'):
            sh_file_path = os.path.join(root, file)
            process_sh_file(sh_file_path)

print("操作完成！")
