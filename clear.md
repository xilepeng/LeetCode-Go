
**Clear Local TimeMachine 备份**

```shell


➜  ~ tmutil listlocalsnapshots /

Snapshots for disk /:
com.apple.TimeMachine.2023-11-11-220005.local
com.apple.TimeMachine.2023-11-12-000607.local


➜  ~ sudo tmutil deletelocalsnapshots 2023-11-11-220005
Password:
Deleted local snapshot '2023-11-11-220005'

➜  ~ sudo tmutil deletelocalsnapshots 2023-11-12-000607
Deleted local snapshot '2023-11-12-000607'
```


**Clear JetBrains Cache**

```shell
rm -rf ~/Library/{Logs/JetBrains &&
rm -rf ~/Library/Application\ Support/JetBrains &&
rm -rf ~/Library/Caches/JetBrains &&
rm -rf ~/Library/Preferences/jetbrains* &&
rm -rf ~/Library/Preferences/com.jetbrains*


brew uninstall --force yarn

rm -rf /Users/x/Library/Caches/Yarn
rm -rf /usr/local/Yarn
rm -rf /usr/local/lib/node_modules/yarn
rm -rf /usr/local/bin/yarn yarnpkg

npm uninstall -g yarn
``` 

