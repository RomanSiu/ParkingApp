from http.client import HTTPException

from fastapi import (APIRouter, Depends, HTTPException, status)
from sqlalchemy.orm import Session

from app.src.database.db import get_db
from app.src.repository.cars import get_car_by_license, ban_car
from app.src.repository.users import get_user_by_id
from app.src.repository import parking as repository_parking
from app.src.schemas import EntryResponse
from app.src.services.auth import RoleChecker
from app.src.database.models import User

router = APIRouter(prefix="/parking", tags=["parking"])


@router.post("/entry")
async def create_entry(car_license: str,
                       db: Session = Depends(get_db)):
    # camera make photo
    # call def with model, to take car license
    car = await get_car_by_license(car_license, db)

    if car is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No car in DB")
    elif car.banned:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Car is blocked")

    last_entry = await repository_parking.get_entry_by_car_id(car.id, db)
    if last_entry.parking_cost == 0:
        await close_entry(car_license, True, db)

    user = await get_user_by_id(car.user_id, db)

    if user.balance <= 0:
        # send message to user phone, about negative balance
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Negative balance")

    entry = await repository_parking.add_entry(user.id, car.id, db)
    return entry


@router.post("/entry/close", response_model=EntryResponse)
async def close_entry(car_license: str = None,
                      free: bool = False,
                      db: Session = Depends(get_db)):
    # if car_license is None:
    #     camera make photo
    #     call def with model, to take car license
    car = await get_car_by_license(car_license, db)
    entry = await repository_parking.get_entry_by_car_id(car.id, db)

    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No entry")

    entry = await repository_parking.close_entry_rep(entry, db)
    await repository_parking.parking_fee(entry, free, db)
    return entry


@router.post("/rate/add")
async def add_new_rate(rate_name: str,
                       price: float,
                       current_user: User = Depends(RoleChecker(allowed_roles=["admin"])),
                       db: Session = Depends(get_db)):
    rate = await repository_parking.find_rate(rate_name, db)

    if rate:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Rate already exist")
    new_rate = await repository_parking.add_rate(rate_name, price, db)
    return new_rate


@router.patch('/cars/ban')
async def ban_unban_user(car_license: str,
                         banned: bool,
                         current_user: User = Depends(RoleChecker(['admin'])),
                         db: Session = Depends(get_db)
                         ) -> dict:
    car = await get_car_by_license(car_license, db)

    if car is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Verification error")

    if car.user_id == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot ban/unban your car")

    message = await ban_car(car, banned, db)
    return {"message": message}
