document.getElementById('cancerForm').addEventListener('submit', function(e) {
    let isValid = true;
    const inputs = this.querySelectorAll('input');

    inputs.forEach(input => {
        const val = input.value.trim();

        // 验证条件：不能为空且必须是数值
        if (val === "" || isNaN(val)) {
            input.classList.add('invalid'); // 输入框变粉红色（通过CSS定义）
            isValid = false;
        } else {
            input.classList.remove('invalid');
        }
    });

    if (!isValid) {
        e.preventDefault(); // 阻止表单发送
        alert("Please enter valid numerical features!");
    }
});