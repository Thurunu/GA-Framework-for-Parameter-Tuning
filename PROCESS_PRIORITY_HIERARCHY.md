# Process Priority Hierarchy for Database Workload

## Problem Identified
MySQL database was configured with `HIGH` priority (-10), but system processes had `CRITICAL` priority (-20), meaning **system processes were getting higher CPU priority than the main database workload**.

## New Priority Hierarchy (Fixed)

### Priority Levels (Nice Values):
```
-20 (CRITICAL)  ← Highest CPU priority
-10 (HIGH)
  0 (NORMAL)
+10 (LOW)
+19 (BACKGROUND) ← Lowest CPU priority
```

### Updated Configuration:

#### 1. **CRITICAL Priority (-20)** - Your Main Workload
```yaml
database:
  priority_class: CRITICAL  # -20
  patterns:
    - mysqld
    - postgres
    - redis
    - mariadb
    - mongodb
```
**Purpose**: Give absolute highest priority to your database workload (MySQL). This ensures database queries and transactions get CPU time first.

#### 2. **HIGH Priority (-10)** - Supporting System Services
```yaml
system_critical:
  priority_class: HIGH  # -10
  patterns:
    - systemd
    - sshd
    - init
    - dbus
```
**Purpose**: System services that support database operations (SSH for remote access, systemd for service management) get second-tier priority.

#### 3. **NORMAL Priority (0)** - General Applications
```yaml
compute_intensive:
  priority_class: NORMAL  # 0
  patterns:
    - python
    - numpy applications
```
**Purpose**: Default priority for compute tasks that don't interfere with database.

#### 4. **LOW Priority (+10)** - Background Compilation
```yaml
compilation:
  priority_class: LOW  # 10
  patterns:
    - gcc
    - make
    - python3 build tasks
```
**Purpose**: Lower priority for build/compilation tasks that can wait.

#### 5. **BACKGROUND Priority (+19)** - Maintenance Tasks
```yaml
background_tasks:
  priority_class: BACKGROUND  # 19
  patterns:
    - cron
    - backup
    - rsync
    - tar
    - gzip
    - logrotate
```
**Purpose**: Lowest priority for maintenance tasks that should never interfere with database operations.

## Expected Behavior After Fix

### Before (Incorrect):
```
PID  USER      PR  NI    COMMAND
739  mysql     10 -10    mysqld        ← Database
1    root       0 -20    systemd       ← System (HIGHER priority!)
```

### After (Correct):
```
PID  USER      PR  NI    COMMAND
739  mysql      0 -20    mysqld        ← Database (HIGHEST priority!)
1    root      10 -10    systemd       ← System (supporting role)
```

## Workload Focus Boost

When `workload_focus='database'` is detected:
```yaml
workload_focus_boost:
  enabled: true
  boost_amount: 5  # Further reduce nice value
  max_priority: -20
```

**Result**: Database processes already at -20 stay at -20 (can't go lower). Other processes get a boost if they match the database workload pattern.

## Testing the Fix

Run your continuous optimizer again:
```bash
python quick_start_continuous.py
```

Expected output:
```
Boosting database process mysqld (PID 739)
Simulated: renice -20 739  ← CRITICAL priority for database

Set process 1 priority to -10    ← HIGH priority for systemd
```

Monitor with `top`:
```bash
top -p $(pgrep -d',' -f 'mysqld|systemd|cron')
```

## Why This Matters

**Database Workload = Your Core Business Logic**
- MySQL handles all data queries
- Needs immediate CPU access for fast query response
- Any CPU contention = slower query times = poor user experience

**System Processes = Supporting Infrastructure**
- Important, but secondary to the main workload
- Can tolerate slight delays (e.g., systemd service checks)
- Should NOT compete with database for CPU

**Background Tasks = Can Wait**
- Log rotation, backups, cron jobs
- Can run during idle periods
- Should be completely deprioritized during heavy database load

## Additional Recommendations

### For Production MySQL Workload:

1. **Enable CPU Affinity** (future enhancement):
   ```bash
   taskset -c 0-3 mysqld  # Pin MySQL to specific CPU cores
   ```

2. **Monitor Priority Changes**:
   ```bash
   watch -n 1 "ps -eo pid,ni,comm | grep -E 'mysqld|systemd|cron'"
   ```

3. **Validate Performance Impact**:
   - Run benchmark before/after priority changes
   - Measure query response times
   - Check CPU utilization distribution

## Files Modified
- `config/process_priorities.yml` - Updated priority mappings:
  - `database`: HIGH (-10) → CRITICAL (-20)
  - `system_critical`: CRITICAL (-20) → HIGH (-10)
  - `background_tasks`: Uncommented and enabled (BACKGROUND = +19)

## Summary

✅ **Fixed**: MySQL now has HIGHEST priority (-20)  
✅ **Fixed**: System processes have SUPPORTING priority (-10)  
✅ **Added**: Background tasks have LOWEST priority (+19)  
✅ **Result**: Database gets CPU priority over everything else

Your main workload (MySQL) now receives the CPU attention it deserves! 🎯
