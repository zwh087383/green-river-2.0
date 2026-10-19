cmake_minimum_required(VERSION 3.8)
project(rag_interfaces)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VoiceQuery.msg"
  "msg/VoiceAnswer.msg"
  "msg/VoiceControl.msg"
  "msg/VoicePlayback.msg"
  "srv/Ask.srv"
  "srv/Search.srv"
  "srv/GetIndexStats.srv"
  "srv/SetDocumentEnabled.srv"
  "srv/SetDisplayMode.srv"
  "action/ParseDocument.action"
  "action/IndexDocument.action"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
