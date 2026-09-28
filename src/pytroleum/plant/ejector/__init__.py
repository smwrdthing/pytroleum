# NOTE Я думаю насчёт того, как привести расчёты любого оборудования к единому виду
# NOTE в абстрактной форме, это должно помочь нам писать код в дальнейшем, на самом
# NOTE верхнем уровне всё будет одинаковым, все детали уйдут в код для конкретного
# NOTE оборудования
# NOTE
# NOTE У нас всегда есть ТЗ/набор ограничений - для этого мы заводим объект-контейнер
# NOTE класса Requirements, при этом оборудование обеспечивает какие-то условия
# NOTE протекания рабочего процесса, для них можно завести объект класса
# NOTE OperationConditions
# NOTE
# NOTE У оборудования есть конструктивные параметры - геометрические размеры, профили
# NOTE проточных частей и т.д., всю эту информацию можно хранить в классе Design
# NOTE
# NOTE В любом расчёте конечные размеры зависят от ТЗ, поэтому мы можем писать
# NOTE очень "высокоуровненвые" функции в форме вроде
# NOTE
# NOTE def design(reuirements: Requirements) -> Design, OperationConditions:
# NOTE      ...
# NOTE      # код для расчёта размеров/параметров тут, определяем конструктивные
# NOTE      # параметры, постепенно заполняем design:Design, также
# NOTE      # заполняем информацию о том, какие рабочие условия устанвоятся
# NOTE      # для оборудования, постепенно заполняем
# NOTE      # operatoin_conditions: OperationConditions
# NOTE      ...
# NOTE      return design, operation_conditions
# NOTE
# NOTE Подход должен получиться гибким и универсальным, для конкретного оборудования нужно
# NOTE будет лишь верно описать классы и функцию design, тогда на самом верхнем уровне
# NOTE останется заполнять параметры из "опросного листа" и вызывать design, например:
# NOTE
# NOTE >> exchanger_requirements = read_from_file(path) # как бы читаем ТЗ из файла
# NOTE >> # Проводим проектный расчёт для теплообменников двух разных видов:
# NOTE >> shell_and_tube_exchanger, shell_and_tube_conditions = \
# NOTE          Exchangers.ShellAndTube.design(exchanger_requirements)
# NOTE >> spiral_exchanger, spiral_conditions = \
# NOTE          Exchangers.Spiral.design(exchanger_requirements)
# NOTE >> # Создаём и выгружаем отчёты
# NOTE >> load(report(shell_and_tube_exchanger, shell_and_tube_conditions))
# NOTE >> load(report(spiral_exchanger, spiral_conditions))
# NOTE
# NOTE На практике сигнатура вызова у design может быть сложнее в зависимости от
# NOTE оборудоавния, можно в целом даже организовывать разные функции с разными
# NOTE процедурами расчёта и собирать их кучкой в один модуль
# NOTE
# NOTE Пока не уверен, следует ли делать функции типа design частью класса Design
# NOTE (или какого-то другого, третьего) или свободными функциями и как быть с
# NOTE OperationConditions и Requirements -у них будут повторяющеся поля, можно
# NOTE развязвть их полностью, либо сделать через наследование, либо попробовать
# NOTE уложиться в один класс с псевдонимом для различия как мы сделали тут
